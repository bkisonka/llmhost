// tokens — именованные API-ключи доступа к /v1 (OpenAI-порту).
// Словарь «имя → токен» в памяти демона, персистится в api_tokens.json рядом
// с данными (атомарная запись, как у config). Управление — lhctl tokens /
// экран F6 интерфейса / control API /v0/tokens; проверка — в proxy.cpp
// наравне с общим ключом openai_auth.
#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace lh::tokens {

// Указать путь к хранилищу и прочитать его (отсутствует — пустой словарь).
void init(const std::string& json_path);

// Сгенерировать новый токен с именем name и сохранить. token — значение
// на выдачу клиенту (оно же хранится — по требованию показывается снова).
bool add(const std::string& name, std::string& token, std::string& err);

// Отозвать токен по имени.
bool remove(const std::string& name, std::string& err);

// Список: [{name, token, created_at, last_used}].
nlohmann::json list();

// Сколько токенов (0 — auth на /v1 не форсируется словарём).
int count();

// Проверить предъявленный токен; при успехе обновляет last_used (в памяти).
bool verify(const std::string& token);

}  // namespace lh::tokens
