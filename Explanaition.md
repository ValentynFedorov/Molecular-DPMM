# Математична формалізація Improved Molecular DPMM (оновлена версія без target_clusters)

## 1. Загальна постановка задачі

Дано набір даних \( \mathbf{X} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\} \), де \( \mathbf{x}_i \in \mathbb{R}^d \).

**Мета**: знайти ефективну кластеризацію \( \mathcal{C} = \{C_1, C_2, \ldots, C_k\} \), де кожен кластер \( C_j \) моделюється гаусівським розподілом із параметрами \( \theta_j = \{\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j, w_j\} \).

## 2. Модель кластера

Кожен кластер \( C_j \) описується як:

$$
p(\mathbf{x} \mid C_j) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)
$$

де:
- \( \boldsymbol{\mu}_j = \frac{1}{|C_j|} \sum_{\mathbf{x} \in C_j} \mathbf{x} \)
- \( \boldsymbol{\Sigma}_j = \frac{1}{|C_j|-1} \sum_{\mathbf{x} \in C_j} (\mathbf{x} - \boldsymbol{\mu}_j)(\mathbf{x} - \boldsymbol{\mu}_j)^T + \lambda \mathbf{I} \)
- \( w_j = \frac{|C_j|}{n} \)

## 3. Функція розкиданості кластера

Зберігається попереднє визначення через:
- коефіцієнт варіації \( CV_j \)
- різницю власних значень \( E_j \)
- комбіновану міру \( S_j \)

## 4. Оцінка якості поділу

Оновлена комбінована оцінка враховує **покращення компактності**, **розділеність**, **баланс**, та **лог-правдоподібність**:

$$
Q_{\text{split}}(C \to C_1, C_2) = 0.3 \cdot \text{CompactnessImprovement} + 0.25 \cdot \min(\text{SeparationScore}, 2.0) + 0.2 \cdot \text{BalanceScore} + 0.25 \cdot \text{LogLikelihoodGain}
$$

### LogLikelihoodGain

$$
\text{LogLikelihoodGain} = \frac{LL(C_1) + LL(C_2) - LL(C)}{|LL(C)| + \epsilon}
$$

де \( LL(C) \) — лог-правдоподібність кластеру через 1-компонентну GMM.

## 5. Критерій поділу

Кластер ділиться, якщо:

$$
Q_{\text{split}}(C_j) > \tau_t
$$

без прив'язки до \( \text{target\_clusters} \). \( \tau_t \) — адаптивний поріг залежно від ітерації.

## 6. Об'єднання кластерів (зворотний прохід)

Залишаються ті самі евристики, але **використовуються всі пари**, і злиття відбувається **до стабілізації**, без обмеження на \( k \).

Оцінка:

$$
Q_{\text{merge}}(C_i, C_j) = 0.5 \cdot (\text{AvgCompactness} - \text{CombinedCompactness}) + 0.5 \cdot \frac{1}{\|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\| + 1}
$$

Об'єднання пари відбувається, якщо \( Q_{\text{merge}} > 0.05 \).

## 7. Класифікація нових точок

Оцінка схожості точки \( \mathbf{x} \) до кластера \( C_j \):

$$
\text{Score}(\mathbf{x}, C_j) = \log p(\mathbf{x} | C_j) + \log w_j
$$

і класифікація через:

$$
\hat{c}(\mathbf{x}) = \arg\max_j \text{Score}(\mathbf{x}, C_j)
$$

## 8. Алгоритм

### Фаза 1: Прямий прохід
1. Ініціалізація (KMeans або один кластер)
2. Ітеративне розбиття кластерів за \( Q_{\text{split}} > \tau \)
3. Refinement через EM

### Фаза 2: Зворотний прохід
1. Обчислення всіх парних \( Q_{\text{merge}} \)
2. Злиття пари з найбільшим \( Q \)
3. Повторення до стабілізації (merge score \( < 0.05 \))

## 9. Метрики якості
- Silhouette
- ARI
- Calinski-Harabasz

## 10. Властивості
- Немає жорсткого обмеження на \( k \)
- Підтримка адаптивних рішень
- Статистичні та евристичні критерії

## 11. Обчислювальна складність

- Прямий прохід: \( O(n \cdot d \cdot k \cdot \text{max\_iter}) \)
- Зворотний прохід: \( O(k^2 \cdot n \cdot d) \) (через всі пари)
- Загальна: \( O(n \cdot d \cdot k (\text{max\_iter} + k)) \)
