// Веб-интерфейс llmhost: один статический HTML-файл, раздаётся control API
// на GET /. Функциональность — главное окно TUI в браузере: модели
// (загрузка/выгрузка/карточка/контекст/MTP/удаление), движок llama.cpp
// (установка сборок, пин, выбор), каталог HuggingFace с поиском и
// скачиванием, токены /v1, настройки [general], чат на нативном SSE-стриме.
//
// Философия файла (обсуждение 10.10): инструмент, а не витрина. Ноль
// зависимостей, ноль сборки, ноль нового бэкенда — прямой fetch к
// существующим /v0/*. Токен UI спрашивает один раз, хранит в localStorage
// и прикладывает к каждому запросу; сам файл секретов не содержит и
// раздаётся без auth — без токена он видит только свои 401. Все серверные
// данные рендерятся через esc(): имена моделей приходят с файловой
// системы. Чат здесь — пятая копия SSE-разбора (после lh::sse::Chat,
// tui, lhctl; chatui — Windows) — осознанная плата за браузер: там парсер
// даёт сама платформа.
//
// Правила файла: ванильный JS без фреймворков и шага сборки; таблицы
// перерисовываются только при изменении данных (иначе киляется фокус
// ввода); новая «фича» — колонка/кнопка на существующем /v0/*-маршруте.
#pragma once

namespace lh {

inline const char* const WEBUI_HTML = R"html(<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llmhost</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 system-ui, "Segoe UI", sans-serif;
       background: #131417; color: #d7d7da; }
header { position: sticky; top: 0; z-index: 5; background: rgba(19,20,23,.94);
         border-bottom: 1px solid #2a2b30; padding: 8px 16px; }
h1 { font-size: 15px; margin: 0; display: inline; }
#sub { color: #8b8d92; font-size: 12.5px; margin-top: 2px; }
main { max-width: 1180px; margin: 0 auto; padding: 10px 16px 60px; }
#tabs { display: flex; gap: 2px; margin: 2px 0 0; border-bottom: 1px solid #2a2b30; }
#tabs button { background: none; border: 1px solid transparent; border-bottom: none;
               border-radius: 6px 6px 0 0; padding: 5px 14px; color: #9aa0a6;
               font-size: 13px; }
#tabs button:hover { background: #1f2126; color: #d7d7da; }
#tabs button.on { background: #1a1b1f; border-color: #2a2b30; color: #f0f0f2; }
section[hidden] { display: none; }
table { border-collapse: collapse; width: 100%; background: #1a1b1f;
        border: 1px solid #2a2b30; border-radius: 6px; overflow: hidden; }
th, td { padding: 5px 9px; text-align: left; border-bottom: 1px solid #232428;
         font-size: 13px; vertical-align: top; }
th { background: #1f2126; color: #9aa0a6; font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: .4px; }
tbody tr:hover { background: #202227; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
button { font: inherit; font-size: 12.5px; background: #26282e; color: #d7d7da;
         border: 1px solid #33353c; border-radius: 4px; padding: 2px 9px;
         cursor: pointer; }
button:hover { background: #2f323a; }
button.primary { background: #2d5a92; border-color: #2d5a92; }
button.danger { background: #5a2d2d; border-color: #6e3a3a; color: #f0b9b9; }
button:disabled { opacity: .45; cursor: default; }
input, select { font: inherit; font-size: 13px; background: #1f2126;
                color: #d7d7da; border: 1px solid #33353c; border-radius: 4px;
                padding: 3px 7px; }
.ok { color: #7ec98a; } .warn { color: #e0c06a; } .err { color: #e88374; }
.dim { color: #8b8d92; }
.badge { background: #23252b; border: 1px solid #33353c; border-radius: 9px;
         padding: 0 7px; font-size: 11.5px; color: #b8bcc4; white-space: nowrap; }
.bar { height: 8px; background: #242630; border-radius: 4px; overflow: hidden;
       min-width: 90px; }
.bar div { height: 100%; background: #4f8dd6; }
.bar div.hot { background: #e0a458; } .bar div.full { background: #e06c5a; }
.cards { display: flex; flex-wrap: wrap; gap: 8px; }
.card { border: 1px solid #2a2b30; background: #1a1b1f; border-radius: 6px;
        padding: 8px 12px; min-width: 300px; flex: 1; }
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
       margin: 6px 0; }
#msg { position: sticky; top: 46px; z-index: 4; min-height: 0; }
#msg div { margin: 6px 0; padding: 5px 10px; border-radius: 5px;
           background: #23252b; border: 1px solid #33353c; }
#msg div.err { border-color: #6e3a3a; }
.modal { position: fixed; inset: 0; background: rgba(0,0,0,.6); z-index: 20;
         display: flex; align-items: flex-start; justify-content: center;
         padding: 40px 12px; overflow: auto; }
.modal .box { background: #1a1b1f; border: 1px solid #33353c; border-radius: 8px;
              padding: 14px 18px; max-width: 760px; width: 100%; }
.box h3 { margin: 0 0 10px; font-size: 15px; }
.kv { display: grid; grid-template-columns: 170px 1fr; gap: 2px 10px;
      font-size: 13px; }
.kv b { color: #9aa0a6; font-weight: 500; }
#chatlog { border: 1px solid #2a2b30; background: #1a1b1f; border-radius: 6px;
           padding: 10px; height: 300px; overflow-y: auto; margin: 6px 0; }
#chatlog .u { color: #c9cdd4; margin: 6px 0; }
#chatlog .a { color: #d7d7da; margin: 6px 0; white-space: pre-wrap; }
#chatlog .r { color: #8b8d92; font-style: italic; white-space: pre-wrap; }
#chatlog .e { color: #e88374; }
</style></head><body>
<header><h1>llmhost</h1> <span class="dim" id="ver"></span>
  <div id="sub"></div></header>
<main>
<div id="msg"></div>
<nav id="tabs">
  <button data-tab="models" class="on">Модели</button>
  <button data-tab="image">Изображения</button>
  <button data-tab="engine">Движок</button>
  <button data-tab="catalog">Каталог</button>
  <button data-tab="chat">Чат</button>
  <button data-tab="tokens">Токены</button>
  <button data-tab="settings">Настройки</button>
</nav>
<section id="tab-models">
<div class="cards" id="cards"></div>
<table id="models"><tbody></tbody></table>
<div class="row">
  <button id="unloadall">Выгрузить всё</button>
  <span class="dim">Карточка — по кнопке «…»; удаление — с подтверждением</span>
</div>
</section>
<section id="tab-engine" hidden>
<div id="engine"></div>
</section>
<section id="tab-image" hidden>
<div class="cards" id="imagecards"></div>
<div class="row">
  <select id="imgmodel"></select>
  <button id="imgload" class="primary">Загрузить</button>
  <button id="imgunload">Выгрузить</button>
  <span class="dim">генерация — /v1/images/* OpenAI-порта; модель меняется без респавна</span>
</div>
</section>
<section id="tab-catalog" hidden>
<div class="row">
  <input id="q" size="28" placeholder="поиск: qwen3, gemma…">
  <button id="searchbtn" class="primary">Найти</button>
  <span class="dim">или скачать сразу:</span>
  <input id="repo" size="24" placeholder="qwen3 или org/repo">
  <input id="file" size="18" placeholder="файл/wildcard (опц.)">
  <button id="pullbtn">Скачать</button>
</div>
<div id="searchout"></div>
<div id="pull" class="dim"></div>
</section>
<section id="tab-chat" hidden>
<div class="row">
  <select id="chatmodel"></select>
  <button id="chatsend" class="primary">Отправить</button>
  <button id="chatstop" disabled>Стоп</button>
  <button id="chatclear">Очистить</button>
</div>
<div id="chatlog"></div>
<div class="row"><input id="chatinput" style="flex:1" placeholder="вопрос…">
</div>
</section>
<section id="tab-tokens" hidden>
<table id="tokens"><tbody></tbody></table>
<div class="row"><input id="tokname" size="14" placeholder="имя">
<button id="tokadd">Выдать</button>
<span class="dim">пока токены есть — /v1 только по ним</span></div>
</section>
<section id="tab-settings" hidden>
<div id="settings"></div>
</section>

<div id="modalhost"></div>
</main>
<script>
'use strict';
var TOKEN = localStorage['lh.token'];
var $ = function (id) { return document.getElementById(id); };
var esc = function (s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
  });
};
var GB = function (b) {
  if (b == null) return '—';
  if (b >= 1073741824) return (b / 1073741824).toFixed(1) + ' ГБ';
  if (b >= 1048576) return (b / 1048576).toFixed(0) + ' МБ';
  if (b >= 1024) return (b / 1024).toFixed(0) + ' КБ';
  return b + ' Б';
};
var STATE = {
  loaded: ['в памяти', 'ok'], loading: ['грузится…', 'warn'],
  unloaded: ['—', 'dim'], asleep: ['спит', 'dim'], failed: ['не загрузилась', 'err']
};
var stateCell = function (st, failed) {
  var s = STATE[st] || (failed ? STATE.failed : STATE.unloaded);
  return '<span class="' + s[1] + '">' + s[0] + '</span>';
};
var msgTimer = null;
function say(t, cls) {
  $('msg').innerHTML = t ? '<div class="' + (cls || '') + '">' + esc(t) + '</div>' : '';
  if (t && cls !== 'err') {
    if (msgTimer) clearTimeout(msgTimer);
    msgTimer = setTimeout(function () { $('msg').innerHTML = ''; }, 6000);
  }
}
function api(method, path, body) {
  return fetch(path, {
    method: method,
    headers: Object.assign(
      { Authorization: 'Bearer ' + TOKEN },
      body ? { 'Content-Type': 'application/json' } : {}),
    body: body ? JSON.stringify(body) : undefined
  }).then(function (r) {
    if (r.status === 401) {
      localStorage.removeItem('lh.token');
      location.reload();
      throw new Error('токен сброшен');
    }
    return r.json().then(function (j) {
      if (!r.ok) throw new Error((j.error && j.error.message) || ('HTTP ' + r.status));
      return j;
    });
  });
}

// ---- модальные окна ----------------------------------------------------------
function modal(html) {
  closeModal();
  var host = document.createElement('div');
  host.className = 'modal';
  host.id = 'modal';
  host.innerHTML = '<div class="box">' + html + '</div>';
  host.addEventListener('click', function (ev) {
    if (ev.target === host) closeModal();
  });
  $('modalhost').appendChild(host);
  return host;
}
function closeModal() {
  var m = $('modal');
  if (m) m.remove();
}

// ---- модели ------------------------------------------------------------------
var lastModelsJson = '', lastStJson = '';
function modelActions(m, loaded) {
  return '<button data-act="' + (loaded ? 'unload' : 'load') + '" data-id="' +
      esc(m.id) + '">' + (loaded ? 'Выгрузить' : 'Загрузить') + '</button> ' +
    '<button data-act="card" data-id="' + esc(m.id) + '">…</button> ' +
    '<button data-act="ctx" data-id="' + esc(m.id) + '">Контекст</button> ' +
    '<button data-act="mtp" data-id="' + esc(m.id) + '">MTP</button> ' +
    '<button class="danger" data-act="rm" data-id="' + esc(m.id) +
    '" data-path="' + esc(m.path) + '">Удалить</button>';
}
function renderModels(md) {
  var s = JSON.stringify(md);
  if (s === lastModelsJson) return;
  lastModelsJson = s;
  var rows = (md.models || []).map(function (m) {
    var loaded = m.state === 'loaded' || (!m.state && m.loaded);
    var fit = m.fit || {};
    var plan = fit.vram_needed
      ? GB(fit.vram_needed) + (fit.fits ? '' : ' <span class="warn">не влезает целиком</span>')
      : '—';
    var ctx = m.ctx_override ? 'ручной ' + m.ctx_override
            : fit.ctx ? 'план ' + fit.ctx : '—';
    var mtp = m.mtp_override ? m.mtp_override
            : (m.mtp_mode && m.mtp_mode !== 'auto') ? 'глоб. ' + m.mtp_mode : '';
    return '<tr><td>' + esc(m.id || m.name) +
      (m.embedding ? ' <span class="badge">эмбеддер</span>' : '') +
      '</td><td>' + stateCell(m.state, m.failed) +
      '</td><td class="num">' + (m.gpu != null && m.gpu >= 0 ? m.gpu : '—') +
      '</td><td class="num">' + ctx +
      '</td><td class="num">' + plan +
      '</td><td class="num">' + GB(m.size) +
      '</td><td>' + (mtp ? '<span class="badge">MTP ' + esc(mtp) + '</span>' : '') +
      '</td><td>' + modelActions(m, loaded) + '</td></tr>';
  });
  $('models').querySelector('tbody').innerHTML =
    '<tr><th>Модель</th><th>Состояние</th><th>Карта</th><th>Контекст</th>' +
    '<th>План VRAM</th><th>Размер</th><th>MTP</th><th>Действия</th></tr>' +
    rows.join('');
}
$('models').addEventListener('click', function (ev) {
  var b = ev.target.closest('button');
  if (!b || !b.dataset.act) return;
  var id = b.dataset.id, act = b.dataset.act;
  if (act === 'load') {
    say('загрузка «' + id + '»…');
    api('POST', '/v0/load', { name: id })
      .then(function () { say('«' + id + '» — загрузка пошла'); })
      .catch(function (e) { say(e.message, 'err'); });
  } else if (act === 'unload') {
    api('POST', '/v0/unload', { model: id })
      .then(function () { say('«' + id + '» выгружена'); })
      .catch(function (e) { say(e.message, 'err'); });
  } else if (act === 'ctx') {
    ctxModal(id);
  } else if (act === 'mtp') {
    mtpModal(id);
  } else if (act === 'card') {
    cardModal(id);
  } else if (act === 'rm') {
    rmModal(id, b.dataset.path);
  }
});
$('unloadall').onclick = function () {
  api('POST', '/v0/unload', {})
    .then(function () { say('выгружено всё свободное'); })
    .catch(function (e) { say(e.message, 'err'); });
};

function findModel(id) {
  var ms = (cache.models && cache.models.models) || [];
  for (var i = 0; i < ms.length; ++i)
    if (ms[i].id === id || ms[i].name === id) return ms[i];
  return null;
}
function ctxModal(id) {
  var m = findModel(id) || {};
  var cur = m.ctx_override || 0, plan = (m.fit || {}).ctx || 0;
  var h = modal('<h3>Контекст «' + esc(id) + '»</h3>' +
    '<div class="kv"><b>Сейчас</b><span>' +
    (cur ? 'ручной ' + cur : 'умная настройка решает сама') +
    (plan ? ' (план ' + plan + ')' : '') + '</span></div>' +
    '<div class="row"><input id="ctxval" size="12" placeholder="' +
    (cur || 'токены, 256…2097152') + '">' +
    '<button class="primary" id="ctxset">Применить</button>' +
    '<button id="ctxauto">Снять (auto)</button>' +
    '<button id="ctxclose">Отмена</button></div>' +
    '<div class="dim">записывается в llmhost.conf, пресет пересобирается сразу</div>');
  h.querySelector('#ctxset').onclick = function () {
    var v = parseInt(h.querySelector('#ctxval').value, 10);
    if (!v || v < 256 || v > 2097152) { say('контекст: 256…2097152', 'err'); return; }
    api('POST', '/v0/models/ctx', { path: m.path, ctx: v })
      .then(function () { closeModal(); say('ctx-size = ' + v + ' записан'); tick(); })
      .catch(function (e) { say(e.message, 'err'); });
  };
  h.querySelector('#ctxauto').onclick = function () {
    api('POST', '/v0/models/ctx', { path: m.path, ctx: 0 })
      .then(function () { closeModal(); say('ручной контекст снят'); tick(); })
      .catch(function (e) { say(e.message, 'err'); });
  };
  h.querySelector('#ctxclose').onclick = closeModal;
}
function mtpModal(id) {
  var m = findModel(id) || {};
  var cur = m.mtp_override || 'auto';
  var h = modal('<h3>MTP-режим «' + esc(id) + '»</h3>' +
    '<div class="kv"><b>Сейчас</b><span>' + esc(cur) +
    (m.mtp_mode ? ' (глобальный: ' + esc(m.mtp_mode) + ')' : '') +
    '</span></div><div class="row">' +
    '<button data-m="auto">auto</button>' +
    '<button data-m="on">on</button>' +
    '<button data-m="off">off</button>' +
    '<button id="mtpclose">Отмена</button></div>' +
    '<div class="dim">auto — решает умная настройка; слот уйдёт в сон и ' +
    'поднимется с новым режимом</div>');
  h.querySelectorAll('button[data-m]').forEach(function (b) {
    b.onclick = function () {
      api('POST', '/v0/models/mtp', { path: m.path, mode: b.dataset.m })
        .then(function () { closeModal(); say('mtp-mode = ' + b.dataset.m); tick(); })
        .catch(function (e) { say(e.message, 'err'); });
    };
  });
  h.querySelector('#mtpclose').onclick = closeModal;
}
function cardModal(id) {
  var m = findModel(id) || {};
  var fit = m.fit || {};
  var args = '';
  var comp = (m.companions || []).map(function (c) {
    return '<div>' + esc(c.kind) + ': ' + esc(c.path) + ' (' + GB(c.size) + ')</div>';
  }).join('');
  var h = modal('<h3>' + esc(m.id || m.name) + '</h3>' +
    '<div class="kv">' +
    '<b>Файл</b><span>' + esc(m.path) + '</span>' +
    '<b>Размер</b><span>' + GB(m.size) + '</span>' +
    '<b>Архитектура</b><span>' + esc(m.arch) + ' · ' + esc(m.quant) + '</span>' +
    '<b>Слои / контекст обучения</b><span>' + (m.n_layers || 0) + ' / ' +
      (m.n_ctx_train ? GB(m.n_ctx_train) + ' ток.' : '—') + '</span>' +
    '<b>Состояние</b><span>' + stateCell(m.state, m.failed) +
      (m.gpu != null && m.gpu >= 0 ? ', карта ' + m.gpu : '') + '</span>' +
    '<b>План VRAM</b><span>' + (fit.vram_needed ? GB(fit.vram_needed) : '—') +
      ', контекст ' + (fit.ctx || '—') + ' (макс ' + (fit.max_ctx || '—') + ')</span>' +
    '<b>Ручной контекст</b><span>' + (m.ctx_override || '—') + '</span>' +
    '<b>MTP</b><span>' + esc(m.mtp_override || m.mtp_mode || 'auto') + '</span>' +
    (comp ? '<b>Спутники</b><span>' + comp + '</span>' : '') +
    '</div><div class="row"><button id="cclose">Закрыть</button></div>');
  h.querySelector('#cclose').onclick = closeModal;
}
function rmModal(id, path) {
  var m = findModel(id) || {};
  var comp = (m.companions || []).map(function (c) {
    return '<div class="dim">' + esc(c.kind) + ': ' + esc(c.path) + '</div>';
  }).join('');
  var total = (m.size || 0) + (m.companions_bytes || 0);
  var h = modal('<h3 class="err">Удалить «' + esc(id) + '» безвозвратно?</h3>' +
    '<div class="kv"><b>Файл</b><span>' + esc(path) + '</span>' +
    '<b>Освободится</b><span>' + GB(total) + '</span></div>' +
    (comp ? '<div class="dim">вместе со спутниками:</div>' + comp : '') +
    '<div class="row"><button class="danger" id="rmgo">Удалить</button>' +
    '<button id="rmno">Отмена</button></div>');
  h.querySelector('#rmgo').onclick = function () {
    api('DELETE', '/v0/models', { path: path })
      .then(function (j) {
        closeModal();
        say('удалено: файлов ' + (j.files || []).length + ', освобождено ' + GB(j.bytes));
        tick();
      })
      .catch(function (e) { say(e.message, 'err'); });
  };
  h.querySelector('#rmno').onclick = closeModal;
}

// ---- карты и шапка -----------------------------------------------------------
var bar = function (used, total) {
  var frac = total ? used / total : 0;
  var cls = frac > 0.9 ? 'full' : frac > 0.75 ? 'hot' : '';
  return '<div class="bar"><div class="' + cls + '" style="width:' +
    Math.min(100, Math.round(frac * 100)) + '%"></div></div>';
};
function gpuCard(c) {
  var used = c.vram_total ? c.vram_total - c.vram_free : 0;
  return '<div class="card"><b>' + esc(c.name || ('карта ' + c.index)) + '</b>' +
    ' <span class="dim">' + GB(used) + ' / ' + GB(c.vram_total) + ' VRAM</span>' +
    bar(used, c.vram_total) +
    '<div class="dim">RAM ' + GB(c.ram_total - c.ram_avail) + ' / ' +
    GB(c.ram_total) + '</div></div>';
}
function renderHeader(st) {
  $('ver').textContent = st.version || '';
  var eng = st.engine || {};
  $('sub').textContent =
    'harness · ' + st.state + ' · движок ' + (eng.build || '—') +
    (eng.selected ? ' (выбрана ' + eng.selected + ')' : '') +
    ' · выгрузка по простою ' + (st.unload_after_min || 0) + ' мин' +
    (st.queue_depth ? ' · очередь ' + st.queue_depth : '') +
    ' · busy ' + (st.busy ? 'да' : 'нет');
  var cards = (st.gpus && st.gpus.length ? st.gpus : [st.gpu]).filter(Boolean);
  $('cards').innerHTML = cards.map(gpuCard).join('');
}

// ---- движок ------------------------------------------------------------------
var enginePct = '';
function renderEngine(st) {
  var eng = st.engine || {};
  var s = JSON.stringify(eng) + (st.busy ? 'b' : '');
  if (s === lastStJson) return;
  lastStJson = s;
  api('GET', '/v0/engine').then(function (j) {
    var pct = '';
    if (j.state === 'downloading' || j.state === 'extracting') {
      var frac = j.total ? j.done / j.total : 0;
      pct = bar(j.done, j.total) + ' <span class="dim">' + GB(j.done) + ' / ' +
            GB(j.total) + '</span>';
    }
    var html = '<div class="card">' +
      '<div class="row"><b>Пин</b> ' + esc(j.release || '?') +
      ' · активная <b>' + esc(j.active || '—') + '</b>' +
      (j.selected ? ' · выбрана ' + esc(j.selected) +
        ' <button data-e="useauto">сбросить</button>' : '') + '</div>' +
      (j.state && j.state !== 'idle'
        ? '<div class="row warn">' + esc(j.state) + ' «' + esc(j.build) + '» ' + pct + '</div>' : '') +
      (j.error ? '<div class="err">' + esc(j.error) + '</div>' : '') +
      (j.mismatch ? '<div class="warn">пин и установленный движок расходятся</div>' : '') +
      '<table><tr><th>Сборка</th><th>Размер</th><th>Статус</th><th></th></tr>' +
      (j.builds || []).map(function (b) {
        var stat = b.installed ? '<span class="ok">установлена</span>' +
            (b.installed_release
              ? (b.installed_release === j.release
                  ? ' <span class="dim">' + esc(b.installed_release) + '</span>'
                  : ' <span class="warn">' + esc(b.installed_release) + ' ≠ пин</span>')
              : '')
          : b.pinned ? '<span class="dim">пиннута, не установлена</span>'
                     : '<span class="warn">без пина</span>';
        return '<tr><td>' + esc(b.id) + '</td><td class="num">' + GB(b.size) +
          '</td><td>' + stat +
          (b.id === j.active ? ' <span class="badge">работает</span>' : '') +
          '</td><td>' + (b.installed ? '' :
            '<button data-e="install" data-b="' + esc(b.id) + '">Установить</button>') +
          (b.id !== j.active ? '<button data-e="use" data-b="' + esc(b.id) + '">Использовать</button>' : '') +
          '</td></tr>';
      }).join('') + '</table>' +
      '<div class="row">' +
      '<button class="primary" data-e="update">Обновить пин + установить</button>' +
      '<button data-e="pinonly">Только пин</button>' +
      '<button data-e="pinrel">Пин конкретного релиза</button></div></div>';
    var host = $('engine');
    host.innerHTML = html;
  }).catch(function (e) { $('engine').innerHTML = '<span class="err">' + esc(e.message) + '</span>'; });
}
$('engine').addEventListener('click', function (ev) {
  var b = ev.target.closest('button');
  if (!b || !b.dataset.e) return;
  var act = b.dataset.e;
  if (act === 'use') {
    api('POST', '/v0/engine/build', { build: b.dataset.b })
      .then(function (j) { say('выбрана сборка ' + j.selected + ' — дети перезапускаются'); tick(); })
      .catch(function (e) { say(e.message, 'err'); });
  } else if (act === 'useauto') {
    api('POST', '/v0/engine/build', { build: 'auto' })
      .then(function () { say('выбор снят — сборка по железу'); tick(); })
      .catch(function (e) { say(e.message, 'err'); });
  } else if (act === 'install') {
    engineInstall(b.dataset.b);
  } else if (act === 'update') {
    enginePin('latest', false);
  } else if (act === 'pinonly') {
    enginePin('latest', true);
  } else if (act === 'pinrel') {
    var rel = prompt('Релиз llama.cpp (bNNNNN):');
    if (rel) enginePin(rel.trim(), true);
  }
});
function engineInstall(build) {
  say('установка «' + build + '» пошла');
  api('POST', '/v0/engine', { build: build })
    .then(function () { lastStJson = ''; tick(); })
    .catch(function (e) { say(e.message, 'err'); });
}
function enginePin(rel, onlyPin) {
  api('POST', '/v0/engine/pin', { release: rel })
    .then(function (j) {
      say('пин: ' + j.release + (onlyPin ? ' (без установки)' : ' — ставлю активную сборку'));
      if (onlyPin) { tick(); return; }
      return api('GET', '/v0/engine').then(function (eng) {
        if (eng.active) return engineInstall(eng.active);
      });
    })
    .catch(function (e) { say(e.message, 'err'); });
}

// ---- каталог и скачивание ----------------------------------------------------
$('searchbtn').onclick = function () {
  var q = $('q').value.trim();
  if (!q) return;
  say('поиск «' + q + '»…');
  api('GET', '/v0/catalog/search?q=' + encodeURIComponent(q) + '&limit=14')
    .then(function (j) {
      say('');
      var rows = (j.results || []).map(function (r) {
        var f = r.fit || {};
        var verdict = !f.known ? '<span class="dim">неизвестно</span>'
          : f.fits ? '<span class="ok">влезает</span>'
          : f.partial || f.dense_partial ? '<span class="warn">частично</span>'
          : '<span class="err">не влезает</span>';
        return '<tr><td>' + esc(r.repo) + '</td>' +
          '<td class="num">' + (f.params ? (f.params / 1e9).toFixed(1) + 'B' : '—') + '</td>' +
          '<td class="num">' + GB(f.vram_needed) + '</td><td>' + verdict +
          '</td><td class="num">' + (r.downloads || 0) + '</td>' +
          '<td><button class="primary" data-pull="' + esc(r.repo) + '">Скачать</button></td></tr>';
      });
      $('searchout').innerHTML = rows.length
        ? '<table><tr><th>Репозиторий</th><th>Параметры</th><th>VRAM</th><th>Вердикт</th>' +
          '<th>Скачиваний</th><th></th></tr>' + rows.join('') + '</table>'
        : '<div class="dim">ничего не нашлось</div>';
    })
    .catch(function (e) { say(e.message, 'err'); });
};
$('searchout').addEventListener('click', function (ev) {
  var b = ev.target.closest('button[data-pull]');
  if (b) startPull(b.dataset.pull, '');
});
$('pullbtn').onclick = function () {
  var repo = $('repo').value.trim();
  if (!repo) { say('укажите имя или org/repo', 'warn'); return; }
  startPull(repo, $('file').value.trim());
};
var pullTick = null;
function startPull(repo, file) {
  var body = { repo: repo };
  if (file) body.file = file;
  api('POST', '/v0/pull', body)
    .then(function () {
      say('скачивание «' + repo + '» началось');
      if (!pullTick) pollPull();
    })
    .catch(function (e) {
      // 409 «уже качается» — просто подключаемся к наблюдению
      if (String(e.message).indexOf('качается') >= 0) { if (!pullTick) pollPull(); }
      else say(e.message, 'err');
    });
}
function pollPull() {
  pullTick = setInterval(function () {
    api('GET', '/v0/pull').then(function (j) {
      if (j.state === 'idle') { clearInterval(pullTick); pullTick = null;
        $('pull').textContent = ''; return; }
      if (j.state === 'error') {
        clearInterval(pullTick); pullTick = null;
        say('скачивание: ' + (j.error || 'ошибка'), 'err');
        $('pull').textContent = '';
        return;
      }
      var p = j.total ? Math.round(j.done * 100 / j.total) + '% · ' : '';
      $('pull').textContent = j.state + ' ' + esc(j.file || j.repo || '') + ' ' + p +
        (j.speed ? (j.speed / 1048576).toFixed(1) + ' МБ/с' : '');
      if (j.state === 'done') {
        clearInterval(pullTick); pullTick = null;
        say('✓ скачано: ' + (j.repo || ''));
        $('pull').textContent = '';
        tick();
      }
    }).catch(function () {});
  }, 1000);
}

// ---- токены ------------------------------------------------------------------
function renderTokens() {
  api('GET', '/v0/tokens').then(function (j) {
    var arr = j.tokens || [];
    $('tokens').querySelector('tbody').innerHTML =
      '<tr><th>Имя</th><th>Токен</th><th>Создан</th><th>Использован</th><th></th></tr>' +
      (arr.length ? arr.map(function (t) {
        return '<tr><td>' + esc(t.name) + '</td><td class="dim">' + esc(t.token) +
          '</td><td class="dim">' + esc(t.created_at || '') + '</td>' +
          '<td class="dim">' + esc(t.last_used || '') + '</td>' +
          '<td><button class="danger" data-tok="' + esc(t.name) + '">Отозвать</button></td></tr>';
      }).join('') : '<tr><td colspan="5" class="dim">токенов нет</td></tr>');
  }).catch(function () {});
}
$('tokens').addEventListener('click', function (ev) {
  var b = ev.target.closest('button[data-tok]');
  if (!b) return;
  api('DELETE', '/v0/tokens', { name: b.dataset.tok })
    .then(function () { say('токен «' + b.dataset.tok + '» отозван'); renderTokens(); })
    .catch(function (e) { say(e.message, 'err'); });
});
$('tokadd').onclick = function () {
  var name = $('tokname').value.trim();
  if (!name) { say('укажите имя токена', 'warn'); return; }
  api('POST', '/v0/tokens', { name: name }).then(function (j) {
    $('tokname').value = '';
    renderTokens();
    var h = modal('<h3>Токен «' + esc(j.name) + '» создан</h3>' +
      '<div class="kv"><b>Значение</b><span style="word-break:break-all">' +
      esc(j.token) + '</span></div>' +
      '<div class="dim">клиенты передают его как Bearer-ключ OpenAI-порта; ' +
      'значение показывается снова: lhctl tokens</div>' +
      '<div class="row"><button id="tokok">Закрыть</button></div>');
    h.querySelector('#tokok').onclick = closeModal;
  }).catch(function (e) { say(e.message, 'err'); });
};

// ---- настройки ---------------------------------------------------------------
function renderSettings() {
  api('GET', '/v0/settings').then(function (j) {
    $('settings').innerHTML = '<div class="dim">' + esc(j.conf) + '</div>' +
      '<table><tr><th>Ключ</th><th>Значение</th><th></th></tr>' +
      (j.settings || []).map(function (s) {
        return '<tr><td>' + esc(s.key) +
          (s.restart_required ? ' <span class="badge">после рестарта</span>' : '') +
          '</td><td><input data-k="' + esc(s.key) + '" value="' + esc(s.value) +
          '" style="width:100%"></td>' +
          '<td><button data-set="' + esc(s.key) + '">Сохранить</button></td></tr>';
      }).join('') + '</table>';
  }).catch(function () {});
}
$('settings').addEventListener('click', function (ev) {
  var b = ev.target.closest('button[data-set]');
  if (!b) return;
  var input = $('settings').querySelector('input[data-k="' + b.dataset.set + '"]');
  api('POST', '/v0/settings', { key: b.dataset.set, value: input.value })
    .then(function (j) {
      say(j.key + ' = ' + j.value +
          (j.restart_required ? ' (вступит после рестарта демона)' : ' (применено)'));
      renderSettings();
    })
    .catch(function (e) { say(e.message, 'err'); });
});

// ---- чат ---------------------------------------------------------------------
var chatHistory = [];
var chatCtl = null;
var chatCreds = null;
function chatModelFill(md) {
  var sel = $('chatmodel');
  var cur = sel.value;
  var opts = (md.models || []).filter(function (m) { return !m.embedding; })
    .map(function (m) { return m.id; });
  var same = sel.dataset.sig === opts.join(',');
  if (!same) {
    sel.dataset.sig = opts.join(',');
    sel.innerHTML = opts.map(function (o) {
      return '<option>' + esc(o) + '</option>';
    }).join('');
  }
  if (cur) sel.value = cur;
}
function chatCredsLoad() {
  if (chatCreds) return Promise.resolve(chatCreds);
  return api('GET', '/v0/credentials').then(function (j) {
    chatCreds = j;
    return j;
  });
}
function chatLine(cls, text) {
  var d = document.createElement('div');
  d.className = cls;
  d.textContent = text;
  $('chatlog').appendChild(d);
  $('chatlog').scrollTop = $('chatlog').scrollHeight;
  return d;
}
$('chatclear').onclick = function () {
  chatHistory = [];
  $('chatlog').innerHTML = '';
};
$('chatstop').onclick = function () { if (chatCtl) chatCtl.abort(); };
$('chatsend').onclick = chatSend;
$('chatinput').addEventListener('keydown', function (ev) {
  if (ev.key === 'Enter' && !ev.shiftKey) { ev.preventDefault(); chatSend(); }
});
function chatSend() {
  var q = $('chatinput').value.trim();
  if (!q || chatCtl) return;
  $('chatinput').value = '';
  chatLine('u', 'Вы: ' + q);
  chatHistory.push({ role: 'user', content: q });
  var a = chatLine('a', '…');
  var think = null;
  chatCtl = new AbortController();
  $('chatsend').disabled = true;
  $('chatstop').disabled = false;
  chatCredsLoad().then(function (cr) {
    var body = { messages: chatHistory.slice(-20), stream: true };
    var model = $('chatmodel').value;
    if (model) body.model = model;
    // База OpenAI — от хоста, с которого открыт дашборд: 127.0.0.1 в
    // openai_base_url — взгляд сервера, чужому браузеру он показывает его
    // собственный localhost (порт OpenAI должен быть достижим по этому
    // хосту: openai-bind 0.0.0.0 или проброс порта тем же туннелем).
    // Старый демон без openai_port — прежнее поведение.
    var base = cr.openai_base_url;
    if (cr.openai_port) {
      var h = location.hostname || '127.0.0.1';
      if (h.indexOf(':') >= 0 && h.charAt(0) !== '[') h = '[' + h + ']';
      base = 'http://' + h + ':' + cr.openai_port + '/v1';
    }
    return fetch(base + '/chat/completions', {
      method: 'POST',
      headers: { Authorization: 'Bearer ' + (cr.api_key || TOKEN),
                 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: chatCtl.signal
    });
  }).then(function (r) {
    if (!r.ok) return r.json().then(function (j) {
      throw new Error((j.error && j.error.message) || ('HTTP ' + r.status));
    });
    var reader = r.body.getReader();
    var dec = new TextDecoder();
    var buf = '', answer = '', reason = '', done = false;
    function pump() {
      return reader.read().then(function (chunk) {
        if (chunk.done) { finish(); return; }
        buf += dec.decode(chunk.value, { stream: true });
        var nl;
        while ((nl = buf.indexOf('\n')) >= 0) {
          var line = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          if (line.charAt(line.length - 1) === '\r') line = line.slice(0, -1);
          if (line.indexOf('data: ') !== 0) continue;
          var payload = line.slice(6);
          if (payload === '[DONE]') { done = true; finish(); return; }
          try {
            var d = JSON.parse(payload).choices[0].delta;
            var r2 = d.reasoning_content || '';
            var c = d.content || '';
            if (r2) {
              if (!think) think = chatLine('r', '');
              reason += r2;
              think.textContent = 'мысли: ' + reason;
              $('chatlog').scrollTop = $('chatlog').scrollHeight;
            }
            if (c) {
              answer += c;
              a.textContent = answer;
              $('chatlog').scrollTop = $('chatlog').scrollHeight;
            }
          } catch (err) {}
        }
        if (!done) return pump();
      });
    }
    function finish() {
      chatCtl = null;
      $('chatsend').disabled = false;
      $('chatstop').disabled = true;
      var text = answer || reason;
      if (text) {
        a.textContent = answer || reason;
        chatHistory.push({ role: 'assistant', content: text });
      } else {
        a.className = 'e';
        a.textContent = 'пустой ответ модели';
      }
    }
    return pump();
  }).catch(function (e) {
    chatCtl = null;
    $('chatsend').disabled = false;
    $('chatstop').disabled = true;
    if (e.name !== 'AbortError') {
      a.className = 'e';
      a.textContent = 'ошибка: ' + e.message;
    } else if (a.textContent === '…') {
      a.remove();
    }
  });
}

// ---- табы ---------------------------------------------------------------------
var ACTIVE = localStorage['lh.tab'] || 'models';
var tabsSeen = {};
function switchTab(name) {
  ACTIVE = name;
  localStorage['lh.tab'] = name;
  document.querySelectorAll('#tabs button').forEach(function (b) {
    b.classList.toggle('on', b.dataset.tab === name);
  });
  ['models', 'image', 'engine', 'catalog', 'chat', 'tokens', 'settings'].forEach(function (t) {
    $('tab-' + t).hidden = t !== name;
  });
  // Ленивые загрузки: токены и настройки — при первом входе во вкладку,
  // движок — свежий срез при каждом входе.
  if (name === 'tokens' && !tabsSeen.tokens) renderTokens();
  if (name === 'settings' && !tabsSeen.settings) renderSettings();
  if (name === 'engine') { lastStJson = ''; }
  tabsSeen[name] = true;
  if (name === 'chat') $('chatinput').focus();
}
$('tabs').addEventListener('click', function (ev) {
  var b = ev.target.closest('button[data-tab]');
  if (b) switchTab(b.dataset.tab);
});

// ---- общий тик ---------------------------------------------------------------
var IMG_STATE = { ru: { unloaded: 'не загружена', loading: 'идёт загрузка…',
                        loaded: 'загружена', error: 'ошибка' } };
function renderImage(j) {
  if (!j) return;
  var ru = IMG_STATE.ru[j.state] || j.state;
  var err = j.error ? '<div class="err">' + esc(j.error) + '</div>' : '';
  var eng = j.engine && j.engine.found
    ? esc(j.engine.path)
    : 'не найден (image-engine или engine/sd/)';
  var models = j.models || [];
  var cards = '<div class="card"><h3>sd-server — ' + esc(ru) + '</h3>' +
    (j.state === 'loaded'
      ? '<div class="kv"><b>Модель</b><span>' + esc(j.model) + '</span>' +
        '<b>Порт</b><span>' + esc(j.port) + '</span>' +
        '<b>В генерации</b><span>' + esc(j.in_flight) + '</span></div>'
      : '') +
    '<div class="kv"><b>Карта</b><span>image-gpu=' + esc(j.gpu) +
    ', простой ≤ ' + esc(j.idle_min) + ' мин</span></div>' + err + '</div>';
  for (var i = 0; i < models.length; ++i) {
    var m = models[i];
    cards += '<div class="card"><h3>' + esc(m.name || m.id) + '</h3>' +
      '<div class="kv"><b>id</b><span>' + esc(m.id) + '</span>' +
      '<b>Веса</b><span>' + GB(m.bytes) + '</span>' +
      '<b>Рецепт</b><span>' + esc(m.recipe) + ' · шаги ' + esc(m.steps) +
      ' · CFG ' + esc(m.cfg) + ' · ' + esc(m.sampler) + '</span></div></div>';
  }
  if (!models.length)
    cards += '<div class="card dim">каталог пуст: ' + esc(j.models_dir) +
             ' — нужен llmhost-image.json + recipes/</div>';
  $('imagecards').innerHTML = cards;
  var sel = $('imgmodel'), cur = sel.value;
  var opts = models.map(function (m) {
    return '<option value="' + esc(m.id) + '">' + esc(m.id) + '</option>';
  }).join('');
  var none = models.length ? '' : '<option value="">(каталог пуст)</option>';
  if (sel.innerHTML !== opts + none) sel.innerHTML = opts + none;
  if (cur && models.some(function (m) { return m.id === cur; })) sel.value = cur;
  $('imgload').disabled = !models.length;
}
$('imgload').onclick = function () {
  var id = $('imgmodel').value;
  say(id ? 'загрузка «' + id + '»…' : 'загрузка…');
  api('POST', '/v0/image/load', { model: id })
    .then(function () { say('загрузка пошла'); tick(); })
    .catch(function (e) { say(e.message, 'err'); });
};
$('imgunload').onclick = function () {
  api('POST', '/v0/image/unload', {})
    .then(function () { say('image-движок остановлен'); tick(); })
    .catch(function (e) { say(e.message, 'err'); });
};

var cache = {};
function tick() {
  Promise.all([api('GET', '/v0/status'), api('GET', '/v0/models'), api('GET', '/v0/image')])
    .then(function (r) {
      cache.status = r[0];
      cache.models = r[1];
      renderHeader(r[0]);
      renderModels(r[1]);
      renderImage(r[2]);
      if (ACTIVE === 'engine') renderEngine(r[0]); // тяжёлый срез — только на своей вкладке
      chatModelFill(r[1]);
    })
    .catch(function (e) { if (e && e.message) say(e.message, 'err'); });
}
if (!TOKEN) {
  TOKEN = prompt('Bearer-токен демона:\n(data/token.txt рядом с lhctl или lhctl tokens)', '');
  if (TOKEN) localStorage['lh.token'] = TOKEN;
}
if (TOKEN) {
  switchTab(ACTIVE);
  tick();
  setInterval(tick, 2000);
}
</script></body></html>
)html";

} // namespace lh
