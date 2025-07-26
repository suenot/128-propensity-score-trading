# Глава 107: Методы Propensity Score для Трейдинга

## Обзор

Методы Propensity Score (PSM) — это мощный набор техник каузального вывода, изначально разработанных в биостатистике для оценки причинно-следственных эффектов из наблюдательных данных. В трейдинге эти методы позволяют ответить на критические вопросы: «Каков истинный каузальный эффект торгового сигнала на доходность?» вместо простого выявления корреляций. Оценивая вероятность (propensity) того, что определённое воздействие (например, сигнал на покупку, смена режима или конкретное состояние рынка) происходит при заданных наблюдаемых ковариатах, мы можем создавать сбалансированные группы сравнения и оценивать несмещённые эффекты воздействия.

Основополагающая работа Rosenbaum & Rubin (1983) представила propensity score matching как метод снижения смещения отбора в наблюдательных исследованиях. На финансовых рынках этот подход помогает трейдерам отличать подлинные альфа-сигналы от ложных корреляций, вызванных смешивающими факторами, такими как рыночный режим, условия волатильности или ликвидность.

В этой главе рассматриваются теория propensity scores, их применение к каузальным торговым стратегиям, реализации на Python и Rust, а также практические примеры с данными фондового рынка и криптовалют с биржи Bybit.

## Содержание

1. [Введение в каузальный вывод в трейдинге](#введение-в-каузальный-вывод-в-трейдинге)
2. [Основы Propensity Score](#основы-propensity-score)
3. [Математические основы](#математические-основы)
4. [Методы оценки Propensity Score](#методы-оценки-propensity-score)
5. [Техники сопоставления и взвешивания](#техники-сопоставления-и-взвешивания)
6. [Применение в трейдинге](#применение-в-трейдинге)
7. [Реализация на Python](#реализация-на-python)
8. [Реализация на Rust](#реализация-на-rust)
9. [Практические примеры с данными акций и криптовалют](#практические-примеры-с-данными-акций-и-криптовалют)
10. [Фреймворк бэктестинга](#фреймворк-бэктестинга)
11. [Оценка производительности](#оценка-производительности)
12. [Дальнейшие направления](#дальнейшие-направления)
13. [Литература](#литература)

---

## Введение в каузальный вывод в трейдинге

### Проблема смешивания (confounding)

В традиционном машинном обучении для трейдинга мы часто находим корреляции между признаками и будущей доходностью. Однако корреляция не означает причинно-следственную связь. Рассмотрим типичный сценарий:

- Моментум-сигнал показывает сильную корреляцию с будущей доходностью
- Но моментум работает лучше в режимах низкой волатильности
- В режимах высокой волатильности тот же сигнал терпит неудачу

Истинная взаимосвязь может быть такой:

```
Режим волатильности → Успешность моментум-сигнала
Режим волатильности → Будущая доходность
```

Здесь режим волатильности — это **конфаундер** (смешивающая переменная), который влияет как на воздействие (активацию сигнала), так и на результат (доходность). Без учёта этого наша оценка эффективности сигнала будет смещённой.

### Фреймворк потенциальных исходов

Модель каузальности Рубина (RCM) формализует каузальный вывод:

- Для каждой единицы i существуют два потенциальных исхода: Y_i(1) при воздействии, Y_i(0) без воздействия
- Индивидуальный эффект воздействия: τ_i = Y_i(1) - Y_i(0)
- Фундаментальная проблема: мы наблюдаем только один исход на единицу

**Средний эффект воздействия (ATE)**:

```
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

В терминах трейдинга:
- **Воздействие**: Применение торгового сигнала (например, открытие длинной позиции при signal > threshold)
- **Исход**: Реализованная доходность
- **ATE**: Истинный каузальный эффект следования сигналу

### Зачем Propensity Scores?

Propensity scores решают проблему смешивания путём:
1. Оценки вероятности воздействия при заданных ковариатах: e(X) = P(T=1|X)
2. Использования этой вероятности для создания сбалансированных групп сравнения
3. Оценки эффектов воздействия, несмещённых конфаундингом

---

## Основы Propensity Score

### Определение

Propensity score — это условная вероятность получения воздействия при заданных наблюдаемых ковариатах:

```
e(X) = P(T = 1 | X)
```

Где:
- T ∈ {0, 1} — индикатор воздействия
- X — вектор наблюдаемых ковариат (конфаундеров)
- e(X) — propensity score

### Свойство балансировки

Ключевой теоретический результат (Rosenbaum & Rubin, 1983):

**Если e(X) = e(X'), то P(X | T=1, e(X)) = P(X | T=0, e(X))**

Это означает, что внутри страт, определённых propensity score, распределение ковариат одинаково для единиц с воздействием и без. Это «балансирует» ковариаты, устраняя смещение от конфаундинга.

### Предположения для каузального вывода

1. **Некоррелированность (Ignorability)**: Y(0), Y(1) ⊥ T | X
   - При заданных наблюдаемых ковариатах назначение воздействия эквивалентно случайному

2. **Позитивность (Overlap)**: 0 < P(T=1|X) < 1 для всех X
   - Каждая единица имеет ненулевую вероятность получить любое из воздействий

3. **SUTVA (Stable Unit Treatment Value Assumption)**:
   - Нет интерференции между единицами
   - Нет скрытых вариаций в воздействии

В трейдинге эти предположения означают:
- Все факторы, влияющие и на активацию сигнала, и на доходность, наблюдаемы
- При любом состоянии рынка сигнал может правдоподобно сработать или нет
- Одна сделка не влияет на результат другой (разумно для ликвидных рынков)

---

## Математические основы

### Оценки среднего эффекта воздействия

**Обратное взвешивание по вероятности (IPW)**:

```
ATE_IPW = (1/n) Σ [T_i Y_i / e(X_i) - (1-T_i) Y_i / (1-e(X_i))]
```

Каждое наблюдение взвешивается обратно вероятности получения фактического воздействия.

**Аугментированное IPW (AIPW / Doubly Robust)**:

```
ATE_AIPW = (1/n) Σ [μ_1(X_i) - μ_0(X_i) + T_i(Y_i - μ_1(X_i))/e(X_i) - (1-T_i)(Y_i - μ_0(X_i))/(1-e(X_i))]
```

Где μ_t(X) = E[Y | T=t, X] — модель исхода. AIPW является дважды робастным: состоятельным, если хотя бы одна из моделей (propensity или outcome) специфицирована правильно.

**Оценка сопоставления**:

```
ATE_match = (1/n) Σ [T_i(Y_i - Y_j(i)) + (1-T_i)(Y_j(i) - Y_i)]
```

Где j(i) — сопоставленная единица для i с ближайшим propensity score из противоположной группы воздействия.

### Оценка дисперсии

Для IPW:

```
Var(ATE_IPW) ≈ (1/n) Var[T Y / e(X) - (1-T) Y / (1-e(X))]
```

Bootstrap обычно используется для оценки дисперсии, особенно для оценок сопоставления.

### Субклассификация

Разделение propensity score на K страт и оценка внутристратных эффектов:

```
ATE_strat = Σ_k (n_k/n) [Ȳ_1k - Ȳ_0k]
```

Cochran (1968) показал, что 5 страт устраняют 90% смещения от одной ковариаты.

---

## Методы оценки Propensity Score

### Логистическая регрессия

Классический подход:

```
e(X) = 1 / (1 + exp(-X'β))
```

Подгонка методом максимального правдоподобия. Простой, интерпретируемый, но предполагает линейную связь в log-odds.

### Градиентный бустинг (GBM)

```python
from sklearn.ensemble import GradientBoostingClassifier
ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
ps_model.fit(X, T)
propensity_scores = ps_model.predict_proba(X)[:, 1]
```

Преимущества:
- Улавливает нелинейные связи
- Работает с многомерными ковариатами
- Часто лучше откалиброван для экстремальных propensity

### Нейронные сети

Для сложных, многомерных данных:

```python
class PropensityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

### Обобщённые Propensity Scores (GPS)

Для непрерывных или многозначных воздействий:

```
GPS(t, X) = f(T=t | X)
```

Это расширяет PSM на вопросы типа: «Каков эффект размера позиции на доходность?»

---

## Техники сопоставления и взвешивания

### Сопоставление по ближайшему соседу

Сопоставление каждой единицы с воздействием с ближайшей единицей без воздействия по propensity score:

```python
def nearest_neighbor_match(ps_treated, ps_control, caliper=None):
    matches = []
    for i, ps_t in enumerate(ps_treated):
        distances = np.abs(ps_control - ps_t)
        if caliper is not None:
            valid = distances <= caliper
            if not valid.any():
                continue
            distances = np.where(valid, distances, np.inf)
        j = np.argmin(distances)
        matches.append((i, j))
    return matches
```

Опции:
- **С заменой**: Контрольные единицы могут сопоставляться несколько раз
- **Без замены**: Каждая контрольная используется один раз (снижает смещение, увеличивает дисперсию)
- **Caliper**: Максимально допустимое расстояние для сопоставления

### Ядерное сопоставление

Взвешивание всех контрольных единиц по их расстоянию до каждой единицы с воздействием:

```
Ŷ_0(i) = Σ_j K((e_i - e_j) / h) Y_j / Σ_j K((e_i - e_j) / h)
```

Где K — ядерная функция (например, гауссова), h — ширина окна.

### Огрублённое точное сопоставление (CEM)

1. Огрубление ковариат в бины
2. Точное сопоставление внутри бинов
3. Использование весов для коррекции разных размеров групп

### Обратное взвешивание по вероятности

Веса для каждой единицы:

- С воздействием: w_i = 1 / e(X_i)
- Контрольные: w_i = 1 / (1 - e(X_i))

Для ATT (Average Treatment Effect on the Treated):
- С воздействием: w_i = 1
- Контрольные: w_i = e(X_i) / (1 - e(X_i))

**Стабилизированные веса** для снижения дисперсии:

```
sw_i = T_i * P(T=1) / e(X_i) + (1-T_i) * P(T=0) / (1-e(X_i))
```

### Веса перекрытия (Overlap Weights)

Решают проблему нарушения позитивности, снижая вес экстремальных propensity:

```
w_i = T_i * (1 - e(X_i)) + (1 - T_i) * e(X_i)
```

Это нацелено на средний эффект воздействия среди популяции с перекрытием (ATO).

---

## Применение в трейдинге

### Сценарий 1: Оценка эффективности сигнала

**Вопрос**: Действительно ли наш моментум-сигнал вызывает положительную доходность, или это конфаундинг от рыночных условий?

**Постановка**:
- Воздействие: Моментум-сигнал срабатывает (T=1) или нет (T=0)
- Исход: Доходность следующего периода
- Ковариаты: Волатильность, объём, спред, время дня и т.д.

**Процесс**:
1. Оценить propensity scores: P(сигнал срабатывает | рыночные условия)
2. Сопоставить периоды срабатывания сигнала с похожими периодами без срабатывания
3. Сравнить среднюю доходность в сопоставленных парах
4. Интерпретировать разницу как каузальный эффект сигнала

### Сценарий 2: Оценка стратегии с поправкой на режим

**Вопрос**: Какова истинная производительность нашей стратегии с контролем на рыночный режим?

**Постановка**:
- Воздействие: Стратегия «активна» (генерирует сделки)
- Исход: Дневной P&L
- Ковариаты: Уровень VIX, сила тренда, ротация секторов и т.д.

Это отделяет навык стратегии от удачной экспозиции на режим.

### Сценарий 3: Анализ исполнения сделок

**Вопрос**: Улучшает ли торговля в определённое время качество исполнения?

**Постановка**:
- Воздействие: Сделка исполнена на открытии рынка vs. в другое время
- Исход: Проскальзывание относительно цены прибытия
- Ковариаты: Размер ордера, спред, волатильность, характеристики акции

### Сценарий 4: Анализ криптовалютного рынка

**Вопрос**: Действительно ли on-chain метрики каузально предсказывают доходность, или они смешаны с рыночным сентиментом?

**Постановка**:
- Воздействие: On-chain активность превышает порог
- Исход: 24-часовая форвардная доходность
- Ковариаты: Корреляция с BTC, профиль объёма, ставка финансирования, социальный сентимент

---

## Реализация на Python

### Модель Propensity Score

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

class PropensityScoreEstimator:
    """
    Оценка Propensity Score различными методами.

    Методы:
    - logistic: Логистическая регрессия
    - gbm: Градиентный бустинг
    - neural: Нейронная сеть
    """

    def __init__(self, method: str = 'gbm'):
        self.method = method
        self.model = None

    def fit(self, X: np.ndarray, T: np.ndarray) -> 'PropensityScoreEstimator':
        """Подгонка модели propensity score."""
        if self.method == 'logistic':
            self.model = LogisticRegression(max_iter=1000, C=1.0)
            self.model.fit(X, T)
        elif self.method == 'gbm':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8
            )
            self.model.fit(X, T)
        elif self.method == 'neural':
            self.model = self._fit_neural(X, T)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание propensity scores."""
        if self.method in ['logistic', 'gbm']:
            return self.model.predict_proba(X)[:, 1]
        elif self.method == 'neural':
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                return self.model(X_tensor).numpy().flatten()


class PropensityScoreMatcher:
    """
    Сопоставление по Propensity Score и оценка эффекта воздействия.
    """

    def __init__(self, method: str = 'nearest', caliper: float = None, replacement: bool = True):
        self.method = method
        self.caliper = caliper
        self.replacement = replacement
        self.matches_ = None

    def match(self, propensity_scores: np.ndarray, treatment: np.ndarray):
        """Выполнить сопоставление по propensity score."""
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        ps_treated = propensity_scores[treated_idx]
        ps_control = propensity_scores[control_idx]

        matched_treated = []
        matched_control = []
        available_controls = set(range(len(control_idx)))

        for i, ps_t in enumerate(ps_treated):
            if not available_controls:
                break

            ctrl_list = list(available_controls)
            distances = np.abs(ps_control[ctrl_list] - ps_t)

            if self.caliper is not None:
                valid = distances <= self.caliper
                if not valid.any():
                    continue
                distances = np.where(valid, distances, np.inf)

            best_idx = ctrl_list[np.argmin(distances)]
            matched_treated.append(treated_idx[i])
            matched_control.append(control_idx[best_idx])

            if not self.replacement:
                available_controls.remove(best_idx)

        self.matches_ = (np.array(matched_treated), np.array(matched_control))
        return self.matches_

    def estimate_ate(self, outcomes, propensity_scores, treatment, method='matching'):
        """Оценка среднего эффекта воздействия (ATE)."""
        if method == 'matching':
            return self._ate_matching(outcomes, treatment)
        elif method == 'ipw':
            return self._ate_ipw(outcomes, propensity_scores, treatment)
        elif method == 'aipw':
            return self._ate_aipw(outcomes, propensity_scores, treatment)


class CausalTradingStrategy:
    """
    Торговая стратегия, использующая методы propensity score
    для оценки и корректировки эффективности сигналов.
    """

    def __init__(self, ps_method='gbm', ate_method='aipw', min_ate=0.0):
        self.ps_estimator = PropensityScoreEstimator(method=ps_method)
        self.matcher = PropensityScoreMatcher()
        self.ate_method = ate_method
        self.min_ate = min_ate

    def fit(self, features, signals, returns):
        """Подгонка каузальной торговой стратегии."""
        self.ps_estimator.fit(features, signals)
        ps = self.ps_estimator.predict(features)
        self.matcher.match(ps, signals)
        self.ate_result_ = self.matcher.estimate_ate(returns, ps, signals, method=self.ate_method)
        return self

    def should_trade(self):
        """Определить, имеет ли сигнал статистически значимый положительный эффект."""
        if not hasattr(self, 'ate_result_'):
            return False
        return self.ate_result_['ate'] > self.min_ate and self.ate_result_['ci_lower'] > 0
```

Подробные примеры кода для загрузки данных и бэктестинга доступны в английской версии README.md.

---

## Реализация на Rust

### Структура проекта

```
107_propensity_score_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   ├── propensity.rs
│   │   └── matching.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── features.rs
│   │   └── bybit.rs
│   ├── backtest/
│   │   ├── mod.rs
│   │   └── engine.rs
│   └── trading/
│       ├── mod.rs
│       └── strategy.rs
└── examples/
    ├── basic_propensity.rs
    ├── causal_trading.rs
    └── backtest_strategy.rs
```

### Основная реализация Propensity Score (Rust)

```rust
/// Логистическая регрессия для оценки propensity score.
pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    max_iter: usize,
}

impl LogisticRegression {
    pub fn new(n_features: usize) -> Self {
        LogisticRegression {
            weights: vec![0.0; n_features],
            bias: 0.0,
            learning_rate: 0.01,
            max_iter: 1000,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n = x.len();
        let n_features = self.weights.len();

        for _ in 0..self.max_iter {
            let mut grad_w = vec![0.0; n_features];
            let mut grad_b = 0.0;

            for i in 0..n {
                let pred = self.predict_proba_single(&x[i]);
                let error = pred - y[i];

                for j in 0..n_features {
                    grad_w[j] += error * x[i][j] / n as f64;
                }
                grad_b += error / n as f64;
            }

            for j in 0..n_features {
                self.weights[j] -= self.learning_rate * grad_w[j];
            }
            self.bias -= self.learning_rate * grad_b;
        }
    }

    pub fn predict_proba_single(&self, x: &[f64]) -> f64 {
        let z: f64 = self.weights.iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum::<f64>() + self.bias;
        sigmoid(z)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Сопоставление по Propensity Score для каузального вывода.
pub struct PropensityMatcher {
    caliper: Option<f64>,
    replacement: bool,
}

impl PropensityMatcher {
    pub fn new(caliper: Option<f64>, replacement: bool) -> Self {
        PropensityMatcher { caliper, replacement }
    }

    /// Сопоставить единицы с воздействием с контрольными.
    pub fn match_units(
        &self,
        propensity_scores: &[f64],
        treatment: &[u8],
    ) -> Vec<(usize, usize)> {
        // Реализация сопоставления...
        Vec::new()
    }

    /// Оценить ATE по сопоставленным парам.
    pub fn estimate_ate_matched(
        &self,
        outcomes: &[f64],
        matches: &[(usize, usize)],
    ) -> ATEResult {
        let n = matches.len();
        if n == 0 {
            return ATEResult {
                ate: 0.0,
                se: f64::INFINITY,
                ci_lower: f64::NEG_INFINITY,
                ci_upper: f64::INFINITY,
                n_matched: 0,
            };
        }

        let diffs: Vec<f64> = matches.iter()
            .map(|(t_idx, c_idx)| outcomes[*t_idx] - outcomes[*c_idx])
            .collect();

        let ate = diffs.iter().sum::<f64>() / n as f64;
        let variance = diffs.iter()
            .map(|d| (d - ate).powi(2))
            .sum::<f64>() / (n - 1).max(1) as f64;
        let se = (variance / n as f64).sqrt();

        ATEResult {
            ate,
            se,
            ci_lower: ate - 1.96 * se,
            ci_upper: ate + 1.96 * se,
            n_matched: n,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ATEResult {
    pub ate: f64,
    pub se: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub n_matched: usize,
}
```

Полная реализация на Rust доступна в директории `src/` и описана в английской версии README.md.

---

## Оценка производительности

| Метрика | Описание | Цель |
|---------|----------|------|
| ATE | Средний эффект воздействия (каузальная сила сигнала) | > 0 с ДИ, не включающим 0 |
| Sharpe Ratio | Доходность с поправкой на риск (годовая) | > 1.0 |
| Sortino Ratio | Доходность с поправкой на нижний риск | > 1.5 |
| Max Drawdown | Максимальная просадка | > -20% |
| Win Rate | Доля прибыльных сделок | > 52% |
| Calmar Ratio | Годовая доходность / Макс. просадка | > 0.5 |

### Диагностика баланса ковариат

Перед доверием каузальным оценкам проверьте, что сопоставление достигает баланса:

- Хороший баланс: |SMD| < 0.1 для всех ковариат после сопоставления
- SMD = Standardized Mean Difference (стандартизированная разность средних)

---

## Дальнейшие направления

1. **Непрерывные воздействия**: Расширение до обобщённых propensity scores для оптимизации размера позиции
2. **Double Machine Learning**: Использование ML для обеих моделей (propensity и outcome) с кросс-фиттингом для несмещённой оценки
3. **Инструментальные переменные**: Комбинирование PSM с IV методами для более сильной идентификации
4. **Гетерогенные эффекты воздействия**: Оценка того, как эффективность сигнала варьируется в зависимости от рыночных условий
5. **Синтетический контроль**: Использование propensity scores для построения синтетических бенчмарков для отдельных активов
6. **Мета-обучающие**: Применение T-learner, S-learner, X-learner подходов для гетерогенности эффекта воздействия
7. **Каузальное открытие**: Автоматическая идентификация конфаундеров из данных с помощью алгоритмов каузального открытия

---

## Литература

1. Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.
2. Rosenbaum, P. R., & Rubin, D. B. (1984). Reducing bias in observational studies using subclassification on the propensity score. *Journal of the American Statistical Association*, 79(387), 516-524.
3. Hirano, K., & Imbens, G. W. (2004). The propensity score with continuous treatments. *Applied Bayesian Modeling and Causal Inference*, 226164, 73-84.
4. Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. *Journal of the American Statistical Association*, 89(427), 846-866.
5. Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing data and causal inference models. *Biometrics*, 61(4), 962-973.
6. Imbens, G. W. (2004). Nonparametric estimation of average treatment effects under exogeneity: A review. *Review of Economics and Statistics*, 86(1), 4-29.
7. Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research*, 46(3), 399-424.

---

## Запуск примеров

### Python

```bash
cd 107_propensity_score_trading/python
pip install -r requirements.txt
python model.py          # Тест моделей propensity score
python data_loader.py    # Тест загрузки данных и признаков
python backtest.py       # Запуск примеров бэктестинга
```

### Rust

```bash
cd 107_propensity_score_trading
cargo build
cargo run --example basic_propensity
cargo run --example causal_trading
cargo run --example backtest_strategy
```
