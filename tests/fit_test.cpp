// Тест «влезет ли»: чистая математика — файлы не нужны.
#include "fit.h"

#include <cassert>
#include <cstdio>
#include <vector>

using namespace lh;

static hw::Gpu gpu_with(hw::Build b, uint64_t free_vram, uint64_t total) {
  hw::Gpu g;
  g.nvidia = true;
  g.build = b;
  g.vram_free = free_vram;
  g.vram_total = total;
  return g;
}

int main() {
  // 8B-модель в Q4_K_M: 36 слоёв, GQA 32/8, embd 4096 → голова 128, KV-головы 1024.
  gguf::Meta m;
  m.n_layers = 36;
  m.n_embd = 4096;
  m.n_head = 32;
  m.n_head_kv = 8;
  uint64_t file = 5ull << 30; // ~5 ГБ весов
  hw::Gpu g = gpu_with(hw::Build::Vulkan, 7800ull << 20, 8ull << 30);
  // KV@4096 = 2*36*1024*4096*2 = 576 МиБ; оверхед Vulkan ~90 МБ + 0,25 ГБ буферы.
  fit::Estimate e = fit::estimate(m, file, 4096, g);
  assert(e.kv == 576u << 20);
  assert(e.fits);

  // Тесная карта: 5700 МиБ свободно — при 4096 не влезает (6043 МиБ), при 2048 — да.
  g.vram_free = 5700ull << 20;
  assert(!fit::estimate(m, file, 4096, g).fits);
  assert(fit::estimate(m, file, 2048, g).fits);
  uint32_t mc = fit::max_ctx(m, file, g, 8192);
  assert(mc == 2048);

  // CUDA-бэкенд прожорливее на ту же модель — потолок контекста ниже или равен.
  hw::Gpu gc = gpu_with(hw::Build::Cuda13, 5700ull << 20, 8ull << 30);
  assert(fit::max_ctx(m, file, gc, 8192) <= mc);

  // Совсем маленькая карта: веса не влезают даже при минимуме.
  hw::Gpu small = gpu_with(hw::Build::Vulkan, 2ull << 30, 4ull << 30);
  assert(!fit::estimate(m, file, 512, small).fits);
  assert(fit::max_ctx(m, file, small, 8192) == 0);

  // Нет данных о картах — не влезает никуда, но тотал честный.
  hw::Gpu none{};
  fit::Estimate ne = fit::estimate(m, file, 4096, none);
  assert(!ne.fits && ne.total == e.total);

  // Голов без GQA (n_head_kv == 0 → берём n_head): KV крупнее.
  gguf::Meta old = m;
  old.n_head_kv = 0;
  fit::Estimate oe = fit::estimate(old, file, 4096, g);
  assert(oe.kv > e.kv);

  // KV q8_0: блок 34 байта на 32 элемента (17/16 от «1 байта») — 17/32 от f16,
  // а не ровно половину (преждевременный «1 байт на элемент» недосчитывал ~6%).
  fit::Estimate q = fit::estimate(m, file, 4096, g, /*kv_q8=*/true);
  assert(q.kv * 32 == e.kv * 17);
  assert(q.total < e.total);
  uint32_t mc16 = fit::max_ctx(m, file, g, 32768, false);
  uint32_t mc8 = fit::max_ctx(m, file, g, 32768, true);
  assert(mc8 >= mc16);

  // MoE на CUDA ест compute больше плотных (роутинг экспертов + CUDA-графы,
  // замер 03.10: qwen3moe-30B ~2,6 ГиБ против 1,9 в плотной оценке) — +1 ГиБ
  // к базе; на Vulkan надбавки нет (не замерен).
  gguf::Meta moe = m;
  moe.n_experts = 128;
  moe.n_experts_used = 8;
  hw::Gpu cuda_g = gpu_with(hw::Build::Cuda13, 7800ull << 20, 8ull << 30);
  fit::Estimate ed = fit::estimate(m, file, 8192, cuda_g);
  fit::Estimate em = fit::estimate(moe, file, 8192, cuda_g);
  assert(em.total == ed.total + (1ull << 30));
  assert(fit::estimate(moe, file, 8192, g).total ==
         fit::estimate(m, file, 8192, g).total);

  // Гибридные арки (qwen35 и её MoE-вариант qwen35moe): токенный KV — четверть
  // слоёв, плюс рекуррентное состояние, не зависящее от контекста. Против
  // консервативной полной математики: KV меньше, но state добавляется.
  gguf::Meta hyb = m;
  hyb.arch = "qwen35moe";
  fit::Estimate e_def = fit::estimate(m, file, 65536, cuda_g, true);
  fit::Estimate e_hyb = fit::estimate(hyb, file, 65536, cuda_g, true);
  assert(e_hyb.kv < e_def.kv && e_hyb.kv * 4 == e_def.kv); // kv_frac 0.25
  // State — на рекуррентных слоях: интервал 4 → каждый 4-й слой full-attention,
  // рекуррентных 27 из 36; калибровка 10000 байт на рекуррентный слой×embd
  // (= прежние 7500 по всем слоям измеренной qwen35: 48 рекуррентных из 64).
  assert(e_hyb.state > 0 && e_hyb.state == 27ull * 4096 * 10000);
  assert(e_def.state == 0);
  gguf::Meta hyb35 = hyb;
  hyb35.arch = "qwen35";
  assert(fit::estimate(hyb35, file, 65536, cuda_g, true).total == e_hyb.total);

  // SWA (gemma4): локальные слои держат лишь окно (+паддинг кэша 1024) и
  // имеют свои голову и KV-головы, глобальные — весь контекст. Числа — с
  // реального gemma-4-26B-A4B: 30 слоёв, глобальные 5 (каждый 6-й),
  // локальные 8×256, глобальные 2×512.
  {
    gguf::Meta sw = m;
    sw.n_layers = 30;
    sw.n_head_kv = 8; // первый элемент массива head_count_kv
    sw.n_swa = 1024;
    sw.n_swa_layers = 25;
    sw.n_embd_head_k = 512;
    sw.n_embd_head_k_swa = 256;
    sw.n_head_kv_local = 8;
    sw.n_head_kv_global = 2;
    hw::Gpu g24 = gpu_with(hw::Build::Cuda12, 22ull << 30, 24ull << 30);
    const uint64_t span = 1024 + 1024; // окно + kSwaCachePad
    // q8-кэш: байт на элемент 34/32 → байты = 34×элементы/16 (fixed point).
    fit::Estimate es = fit::estimate(sw, file, 32768, g24, true);
    assert(es.kv == 34ull * (5ull * 32768 * 512 * 2 + 25ull * span * 256 * 8) / 16);
    // Контекст не выше окна: локальные слои — ровно окно, без паддинга сверх.
    fit::Estimate es1 = fit::estimate(sw, file, 1024, g24, true);
    assert(es1.kv == 34ull * (5ull * 1024 * 512 * 2 + 25ull * 1024 * 256 * 8) / 16);
    // Слепая математика (нет окна/паттерна/классов): все 30 слоёв × весь
    // контекст × 512 × 8 голов — переоценка на гигабайты, ради неё всё и
    // затевалось.
    gguf::Meta bl = sw;
    bl.n_swa = 0;
    bl.n_swa_layers = 0;
    bl.n_embd_head_k_swa = 0;
    bl.n_head_kv_local = 0;
    bl.n_head_kv_global = 0;
    fit::Estimate eb = fit::estimate(bl, file, 32768, g24, true);
    assert(eb.kv == 34ull * 30ull * 32768 * 512 * 8 / 16);
    assert(es.kv < eb.kv && eb.kv - es.kv > (3ull << 30));
    // Без явного key_length голова считается из embd (2816/16 = 176), а не
    // из ключа — предпочитается метаданные, fallback остаётся на месте.
    gguf::Meta fk = bl;
    fk.n_embd_head_k = 0;
    fk.n_embd = 2816;
    fk.n_head = 16;
    fit::Estimate ef = fit::estimate(fk, file, 32768, g24, true);
    assert(ef.kv == 34ull * 30ull * 32768 * 176 * 8 / 16);
  }

  // llama4 (iswa): движок для арки сам выводит SWA-геометрию, которой в GGUF
  // нет — окно 8192, каждый 4-й слой глобальный. Числа — с реального
    // unsloth Llama-4-Scout-17B: 48 слоёв (12 глобальных), KV-головы 8,
    // голова 128. Головы у классов одинаковы — key_length_swa не задан,
    // локальный класс берёт общий fallback.
  {
    gguf::Meta l4 = m;
    l4.arch = "llama4";
    l4.n_layers = 48;
    l4.n_head_kv = 8;
    l4.n_embd_head_k = 128;
    l4.n_swa = 8192;
    l4.n_swa_layers = 36;
    hw::Gpu g24 = gpu_with(hw::Build::Cuda12, 22ull << 30, 24ull << 30);
    const uint64_t span = 8192 + 1024; // окно + kSwaCachePad
    // KV 131k q8: глобальные 12 × весь контекст, локальные 36 × окно —
    // ~3.86 ГиБ против «слепых» 48 × весь контекст ≈ 12.75 ГиБ.
    fit::Estimate el = fit::estimate(l4, file, 131074, g24, true);
    assert(el.kv == 34ull * (12ull * 131074 * 128 * 8 + 36ull * span * 128 * 8) / 16);
    gguf::Meta lb = l4;
    lb.n_swa = 0;
    lb.n_swa_layers = 0;
    fit::Estimate eb4 = fit::estimate(lb, file, 131074, g24, true);
    assert(eb4.kv == 34ull * 48ull * 131074 * 128 * 8 / 16);
    assert(eb4.kv - el.kv > (8ull << 30)); // ~9 ГиБ фантомов — ради них и вывод
    // Контекст не выше окна: локальные слои — ровно контекст (без паддинга),
    // форма совпадает со слепой, окно ещё не действует.
    fit::Estimate el1 = fit::estimate(l4, file, 4096, g24, true);
    assert(el1.kv == 34ull * 48ull * 4096 * 128 * 8 / 16);
  }

  // Замеренный оверхед подменяет аналитический: при ctx замера — точно он,
  // выше — растёт ramp'ом (CUDA 16 КиБ/токен), ниже — не ниже базы бэкенда.
  {
    fit::Estimate a = fit::estimate(m, file, 8192, cuda_g, true);
    fit::Measured ms{(int64_t)(600ull << 20), 8192, true};
    fit::Estimate w = fit::estimate(m, file, 8192, cuda_g, true, &ms);
    assert(w.overhead == 600ull << 20);
    assert(w.total == a.total - a.overhead + (600ull << 20));
    fit::Estimate w2 = fit::estimate(m, file, 8192 + 1024, cuda_g, true, &ms);
    assert(w2.overhead == (600ull << 20) + (1024ull << 14));
    fit::Measured tiny{1, 65536, true}; // замер почти в ноль — держим базу
    fit::Estimate w3 = fit::estimate(m, file, 4096, cuda_g, true, &tiny);
    assert(w3.overhead == fit::fixed_overhead(true));
    fit::Measured none{}; // пустой замер — чистая аналитика
    assert(fit::estimate(m, file, 8192, cuda_g, true, &none).total == a.total);
  }

  // MoE-on-CPU: экспертам нужно 85% файла + запас на систему.
  uint64_t ten_gb = 10ull << 30;
  assert(fit::expert_bytes_estimate(ten_gb) == ten_gb / 100 * 85);
  hw::Gpu rich = gpu_with(hw::Build::Vulkan, 0, 0);
  rich.ram_total = 32ull << 30;
  assert(fit::cpu_offload_ok(ten_gb, rich)); // 8,5 ГБ экспертов ≤ 32−4
  hw::Gpu poor = rich;
  poor.ram_total = 8ull << 30;
  assert(!fit::cpu_offload_ok(ten_gb, poor)); // 8,5 > 8−4 — нечего соваться
  hw::Gpu unknown = rich;
  unknown.ram_total = 0;
  assert(!fit::cpu_offload_ok(ten_gb, unknown));

  // Точные байты экспертов (из тензорного списка GGUF) проходят через
  // expert_bytes_of; без них (0) и при мусоре (больше файла) — оценка 85%.
  {
    gguf::Meta ex = moe;
    ex.expert_bytes = (uint64_t)(4.6 * 1073741824.0); // реальная доля MoE ~90%
    assert(fit::expert_bytes_of(ex, file) == ex.expert_bytes);
    gguf::Meta noex = moe;
    assert(fit::expert_bytes_of(noex, file) == fit::expert_bytes_estimate(file));
    gguf::Meta bad = ex;
    bad.expert_bytes = file + 1;
    assert(fit::expert_bytes_of(bad, file) == fit::expert_bytes_estimate(file));
    // Точность решает на границе RAM: оценка 85% (4,25 ГиБ) проходит,
    // реальные 4,6 ГиБ — нет.
    hw::Gpu midram = rich;
    midram.ram_total = (8ull << 30) + (512ull << 20); // бюджет 4,5 ГиБ
    assert(!fit::cpu_offload_ok(ex, file, midram));
    assert(fit::cpu_offload_ok(noex, file, midram));
    assert(fit::cpu_offload_ok(ex, file, rich));
  }

  // Плотный частичный сплит: модель крупнее VRAM — максимум слоёв на GPU.
  // Файл 5 ГБ на 36 слоёв (~142 МиБ/слой с растворёнными emb/lm_head),
  // CUDA-карта с 3,5 ГБ свободно: KV@2048 q8 ~153 МиБ (блок 34/32),
  // оверхед 944 МиБ (400 бэкенд + 512 база FA + ramp 32 МиБ).
  {
    hw::Gpu tight = gpu_with(hw::Build::Cuda13, 3584ull << 20, 8ull << 30);
    fit::DenseSplit s = fit::dense_split(m, file, tight, 32768, /*kv_q8=*/true, 2048);
    // ctx — максимальное КРАТНОЕ 1024 (шаг ответа): 3072 не влезает, 2048 — да.
    assert(s.ok && s.ngl == 17 && s.ctx == 2048);
    assert(s.gpu_bytes == file * 17 / 36 && s.cpu_bytes == file - s.gpu_bytes);
    // При найденном сплите оценка влезает; слоем выше — уже нет.
    assert(fit::estimate(m, s.gpu_bytes, s.ctx, tight, true).fits);
    assert(!fit::estimate(m, file * 18 / 36, s.ctx, tight, true).fits);
    // Меньше минимальный контекст — слоёв на GPU можно больше.
    fit::DenseSplit more = fit::dense_split(m, file, tight, 32768, true, 512);
    assert(more.ok && more.ngl > s.ngl);
    // Крошечная карта: даже бар слоёв не набирает контекст 2048 — отказ.
    hw::Gpu tiny = gpu_with(hw::Build::Cuda13, 1200ull << 20, 4ull << 30);
    assert(!fit::dense_split(m, file, tiny, 32768, true, 2048).ok);
    // Влезающая целиком модель: сплит отдаёт все слои (контракт честный,
    // хотя вызывающие зовут его только для «не влезает»).
    hw::Gpu big = gpu_with(hw::Build::Vulkan, 7800ull << 20, 8ull << 30);
    fit::DenseSplit whole = fit::dense_split(m, file, big, 8192, false, 2048);
    assert(whole.ok && whole.ngl == 36);

    // Экстра-груз на GPU (draft, mmproj — едут целиком, слоями не режутся)
    // сжимает сплит: при том же минимуме контекста слоёв на GPU меньше.
    fit::DenseSplit plain = fit::dense_split(m, file, tight, 32768, true, 2048);
    const uint64_t draft = 600ull << 20;
    fit::DenseSplit with_extra =
        fit::dense_split(m, file, tight, 32768, true, 2048, draft);
    assert(plain.ok && with_extra.ok && with_extra.ngl < plain.ngl);
    // Найденная ступень влезает вместе с грузом; слоем выше — нет.
    assert(fit::estimate(m, with_extra.gpu_bytes + draft, with_extra.ctx, tight,
                         true).fits);
    assert(!fit::estimate(m, file * (with_extra.ngl + 1) / 36 + draft,
                          with_extra.ctx, tight, true).fits);
    // Груз выталкивает за нижний бар слоёв — честный отказ, а не нулевой ngl.
    // Карта ровно под нижний бар: 9 слоёв (1280 МиБ) + KV@2048 q8 (153) +
    // оверхед CUDA (944) = 2377 МиБ → 2400 с запасом; +300 МиБ груза — бар
    // не набирается.
    hw::Gpu floor = gpu_with(hw::Build::Cuda13, 2400ull << 20, 8ull << 30);
    fit::DenseSplit base9 = fit::dense_split(m, file, floor, 32768, true, 2048);
    assert(base9.ok && base9.ngl == fit::dense_min_layers(36) && base9.ctx == 2048);
    assert(!fit::dense_split(m, file, floor, 32768, true, 2048,
                             300ull << 20).ok);
  }

  // MoE частичный сплит (--n-cpu-moe): минимальное N блоков в RAM, дающее
  // контекст не ниже ctx_min. Файл 50 ГиБ: не-экспертная часть 4 ГиБ, блоки
  // 2..47 несут по 1 ГиБ экспертов (блоки 0-1 плотные — first-k-dense).
  // Карта 16 ГиБ: при N=39 на GPU остаётся 13 ГиБ — веса 13312 МиБ +
  // KV q8 @8192 (816 МиБ, блок 34/32) + оверхед 2064 МиБ (с MoE +1 ГиБ) =
  // 16192 ≤ 16384; при N=38 свободно 2 ГиБ — контекст 4096 уже не набирается
  // (максимум 512).
  {
    gguf::Meta mm;
    mm.n_layers = 48;
    mm.n_embd = 4096;
    mm.n_head = 32;
    mm.n_head_kv = 8;
    mm.n_experts = 64;
    mm.n_experts_used = 8;
    mm.expert_bytes = 46ull << 30;
    mm.expert_bytes_by_block.assign(48, 1ull << 30);
    mm.expert_bytes_by_block[0] = 0; // плотные блоки: выносить нечего
    mm.expert_bytes_by_block[1] = 0;
    const uint64_t big_file = 50ull << 30;
    // GPU-часть по точным поблочным байтам: плотные блоки ничего не выносят.
    assert(fit::moe_gpu_bytes(mm, big_file, 0) == big_file);
    assert(fit::moe_gpu_bytes(mm, big_file, 2) == big_file);
    assert(fit::moe_gpu_bytes(mm, big_file, 3) == big_file - (1ull << 30));
    assert(fit::moe_gpu_bytes(mm, big_file, 48) == big_file - (46ull << 30));
    assert(fit::moe_gpu_bytes(mm, big_file, 1000) == big_file - (46ull << 30));

    hw::Gpu card = gpu_with(hw::Build::Cuda13, 16ull << 30, 16ull << 30);
    card.ram_total = 64ull << 30;
    fit::MoESplit sp = fit::moe_split(mm, big_file, card, 8192, /*kv_q8=*/true, 4096);
    assert(sp.ok && sp.ncmoe == 39 && sp.ctx == 8192);
    assert(sp.gpu_bytes == fit::moe_gpu_bytes(mm, big_file, 39));
    assert(sp.cpu_bytes == big_file - sp.gpu_bytes);
    // Минимальность: блоком меньше контекст 4096 уже не набирается.
    assert(fit::max_ctx(mm, fit::moe_gpu_bytes(mm, big_file, 38), card, 8192, true) <
           4096);
    // Найденная ступень живёт в бюджете.
    assert(fit::estimate(mm, sp.gpu_bytes, sp.ctx, card, true).fits);
    // Экстра-груз (draft/mmproj — едут на GPU целиком) выталкивает блок.
    fit::MoESplit ex = fit::moe_split(mm, big_file, card, 8192, true, 4096, 1ull << 30);
    assert(ex.ok && ex.ncmoe == 40);
    // Пин контекста как минимум расклада: тот же N, заказанный контекст.
    fit::MoESplit pin = fit::moe_split(mm, big_file, card, 8192, true, 8192);
    assert(pin.ok && pin.ncmoe == 39 && pin.ctx == 8192);
    // Заказанный контекст недостижим даже при полном выносе — отказ.
    assert(!fit::moe_split(mm, big_file, card, 8192, true, 32768).ok);
    // Без поблочных байт раскладка — догадка: честный отказ.
    gguf::Meta blind = mm;
    blind.expert_bytes_by_block.clear();
    assert(!fit::moe_split(blind, big_file, card, 8192, true, 4096).ok);
    // Тесная RAM: ни одно N не проходит RAM-гейт — отказ, хоть VRAM и влезает.
    hw::Gpu smallram = card;
    smallram.ram_total = 8ull << 30;
    assert(!fit::moe_split(mm, big_file, smallram, 8192, true, 4096).ok);
    // Плотная модель: экспертных байт нет — раскладки не существует.
    gguf::Meta dense = mm;
    dense.expert_bytes_by_block.clear();
    dense.expert_bytes = 0;
    assert(!fit::moe_split(dense, big_file, card, 8192, true, 4096).ok);
  }

  // Гибридность из меты — сильнее таблицы семейств: qwen4exp в таблице не
  // значится, и без ключей планировался как полная attention-модель (KV
  // вчетверо больше реального, состояние не учтено — поймано живьём).
  {
    gguf::Meta q4 = m;
    q4.arch = "qwen4exp";
    fit::Estimate blind = fit::estimate(q4, file, 65536, cuda_g, true);
    assert(blind.state == 0);
    q4.full_attn_interval = 4;
    q4.has_ssm = true;
    fit::Estimate hyb4 = fit::estimate(q4, file, 65536, cuda_g, true);
    assert(hyb4.kv * 4 == blind.kv); // токенный KV на четверти слоёв
    assert(hyb4.state == 27ull * 4096 * 10000); // 27 рекуррентных из 36
    assert(hyb4.total < blind.total);
    // Враждебная мета: интервал больше числа слоёв — clamp до n_layers,
    // иначе недоверенный GGUF обнуляет оценку KV (sat-философия меты).
    gguf::Meta hostile = q4;
    hostile.full_attn_interval = 999999;
    fit::Estimate hh = fit::estimate(hostile, file, 65536, cuda_g, true);
    assert(hh.kv * 36 <= blind.kv * 35 || hh.kv <= blind.kv); // ~1/36, не ~0
    assert(hh.kv > blind.kv / 40);
    // Интервал сильнее таблицы: известная арка с ключом считается по ключу.
    gguf::Meta li = m;
    li.arch = "qwen35moe"; // таблица: тоже 0.25/7500
    li.full_attn_interval = 2;
    li.has_ssm = true;
    fit::Estimate li2 = fit::estimate(li, file, 65536, cuda_g, true);
    assert(li2.kv * 2 == e_def.kv);
    // Интервал 2 → рекуррентных 18 из 36: state масштабируется интервалом
    // меты, а не «все слои» (прежняя математика завышала вдвое).
    assert(li2.state == 18ull * 4096 * 10000);
  }

  // CPU-тензоры (per_layer_token_embd) вычитаются из GPU-части весов.
  {
    gguf::Meta ce = moe;
    ce.expert_bytes = file / 100 * 69;
    ce.cpu_extra_bytes = file / 4;
    assert(fit::vram_weights_of(ce, file) == file - file / 100 * 69 - file / 4);
    // CPU-тензоров больше GPU-части — не уводим в минус, консервативно всё.
    ce.cpu_extra_bytes = file;
    assert(fit::vram_weights_of(ce, file) == file - file / 100 * 69);
    // Без CPU-тензоров — прежняя формула.
    gguf::Meta clean = moe;
    clean.expert_bytes = file / 100 * 90;
    assert(fit::vram_weights_of(clean, file) == file - file / 100 * 90);
    assert(fit::less_cpu_extra(ce, (5ull << 30)) == 5ull << 30);
  }

  // Пул карт: сумма свободной памяти минус оверхед всех карт, кроме первой
  // (estimate добавит свой единожды — итого по одному на каждую).
  {
    std::vector<hw::Gpu> two = {gpu_with(hw::Build::Cuda13, (12ull << 30) - (400ull << 20),
                                         12ull << 30),
                                gpu_with(hw::Build::Cuda13, (24ull << 30) - (400ull << 20),
                                         24ull << 30)};
    hw::Gpu p = fit::pool(two, hw::Build::Cuda13);
    // обе карты в тесте уже «свободны минус оверхед», вычет (N−1) поверх:
    assert(p.vram_free == (36ull << 30) - 3 * (400ull << 20));
    assert(p.vram_total == 36ull << 30);
    // Карта без телеметрии в пул не попадает: планировать по ней нельзя.
    hw::Gpu blind = gpu_with(hw::Build::Cuda13, 8ull << 30, 8ull << 30);
    blind.nvidia = false;
    blind.vram_total = 0;
    assert(fit::pool({blind}, hw::Build::Cuda13).vram_free == 0);
    assert(fit::pool({blind, two[0]}, hw::Build::Cuda13).vram_free == two[0].vram_free);
    assert(fit::pool({}, hw::Build::Cuda13).vram_free == 0);
    // Почти забитые карты: свободная сумма меньше оверхедов соседей —
    // беззнаковое вычитание раньше уезжало вниз («бесконечная» память,
    // все оценки «влезает»); насыщающее вычитание даёт честное «мимо».
    std::vector<hw::Gpu> stuffed = {gpu_with(hw::Build::Cuda13, 100ull << 20, 24ull << 30),
                                    gpu_with(hw::Build::Cuda13, 100ull << 20, 24ull << 30)};
    assert(fit::pool(stuffed, hw::Build::Cuda13).vram_free == 0);
    // Одиночная карта вычета не имеет (n−1 = 0) — свободная как есть.
    assert(fit::pool({stuffed[0]}, hw::Build::Cuda13).vram_free == 100ull << 20);
  }

  // Бисекция max_ctx: между ступенями лестницы теперь не теряется до 2×
  // контекста. Карта, где 49k влезает, а 65536 — нет: прежний ответ с hi 64k
  // был 32768.
  {
    hw::Gpu mid = gpu_with(hw::Build::Cuda13, 11000ull << 20, 12ull << 30);
    uint32_t bm = fit::max_ctx(m, file, mid, 65536, /*kv_q8=*/true);
    assert(bm >= 49152 && bm < 65536);
    assert(bm % 1024 == 0); // кратность шага
    assert(fit::estimate(m, file, bm, mid, true).fits);
    assert(!fit::estimate(m, file, bm + 1024, mid, true).fits);
  }

  // Насыщающая арифметика:Meta-скаляры на своих потолках (65535 слоёв, 2^20
  // размерности) при полном контексте дают произведения за гранью u64 —
  // оценка обязана насытиться в «не влезает никогда», а не замкнуться в
  // маленькое число и выдать ложное fits (план впритык → OOM ребёнка).
  {
    gguf::Meta bad;
    bad.n_layers = 65535;
    bad.n_ctx_train = 1ull << 26; // на потолке gguf::assign
    bad.n_embd = 1ull << 20;
    bad.n_head = 1;
    bad.n_head_kv = 1ull << 20;
    bad.n_embd_head_k = 1ull << 20;
    bad.n_experts = 1ull << 16;
    hw::Gpu big = gpu_with(hw::Build::Cuda13, 1ull << 40, 1ull << 40);
    fit::Estimate b = fit::estimate(bad, 1ull << 20, (uint32_t)(1ull << 26), big);
    assert(b.kv == ~(uint64_t)0); // произведение скаляров насытилось
    assert(b.total == ~(uint64_t)0);
    assert(!b.fits); // ни одна VRAM против насыщения не пройдёт
    // max_ctx на таких метаданных завершается и отказывает; hi ограничен
    // правилом «не выше 32768», а не ctx_train из заголовка.
    assert(fit::max_ctx(bad, 1ull << 20, big, (uint32_t)bad.n_ctx_train) == 0);
  }

  std::puts("fit_test: ok");
  return 0;
}
