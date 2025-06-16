# Математична формалізація Improved Molecular DPMM

## 1. Загальна постановка задачі

Дано набір даних $\mathbf{X} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$, де $\mathbf{x}_i \in \mathbb{R}^d$.

**Мета**: знайти оптимальну кластеризацію $\mathcal{C} = \{C_1, C_2, \ldots, C_k\}$, де кожен кластер $C_j$ характеризується параметрами $\theta_j = \{\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j, w_j\}$.

## 2. Модель кластера

Кожен кластер $C_j$ моделюється багатовимірним нормальним розподілом:

$$p(\mathbf{x} | C_j) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$$

де:
- $\boldsymbol{\mu}_j = \frac{1}{|C_j|} \sum_{\mathbf{x} \in C_j} \mathbf{x}$ — центроїд кластера  
- $\boldsymbol{\Sigma}_j = \frac{1}{|C_j|-1} \sum_{\mathbf{x} \in C_j} (\mathbf{x} - \boldsymbol{\mu}_j)(\mathbf{x} - \boldsymbol{\mu}_j)^T + \lambda \mathbf{I}$ — коваріаційна матриця з регуляризацією  
- $w_j = \frac{|C_j|}{n}$ — вага кластера  
- $\lambda = 10^{-6}$ — параметр регуляризації  

## 3. Функція розкиданості кластера

**Коефіцієнт варіації відстаней:**

$$CV_j = \frac{\sigma_j}{\bar{d}_j + \epsilon}$$

де:
- $\bar{d}_j = \frac{1}{|C_j|} \sum_{\mathbf{x} \in C_j} \|\mathbf{x} - \boldsymbol{\mu}_j\|_2$ — середня відстань до центроїда  
- $\sigma_j = \sqrt{\frac{1}{|C_j|} \sum_{\mathbf{x} \in C_j} (\|\mathbf{x} - \boldsymbol{\mu}_j\|_2 - \bar{d}_j)^2}$ — стандартне відхилення  
- $\epsilon = 10^{-8}$ — параметр стабільності  

**Показник витягнутості (через PCA):**

$$E_j = \lambda_1^{(j)} - \lambda_2^{(j)}$$

де $\lambda_1^{(j)} \geq \lambda_2^{(j)}$ — перші два власні значення $\boldsymbol{\Sigma}_j$.

**Комбінована міра розкиданості:**

$$S_j = 0.7 \cdot CV_j + 0.3 \cdot E_j$$

## 4. Оцінка якості поділу кластера

### 4.1. Покращення компактності

$$\text{CompactnessImprovement} = \frac{\text{Comp}(C) - \frac{|C_1| \cdot \text{Comp}(C_1) + |C_2| \cdot \text{Comp}(C_2)}{|C|}}{\text{Comp}(C) + \epsilon}$$

де:

$$\text{Comp}(C_j) = \frac{1}{|C_j|} \sum_{\mathbf{x} \in C_j} \|\mathbf{x} - \boldsymbol{\mu}_j\|_2$$

### 4.2. Розділеність кластерів

$$\text{SeparationScore} = \frac{\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|_2}{\frac{\text{Comp}(C_1) + \text{Comp}(C_2)}{2} + \epsilon}$$

### 4.3. Збалансованість розмірів

$$\text{BalanceScore} = \frac{\min(|C_1|, |C_2|)}{\max(|C_1|, |C_2|)}$$

### 4.4. Статистична значущість

Для багатовимірних даних використовується t-тест на проєкцію PCA:

$$\text{SignificanceScore} = \max(0, 1 - p\text{-value})$$

### 4.5. Комбінована оцінка якості поділу

$$Q_{\text{split}}(C \to C_1, C_2) = 0.3 \cdot \text{CompactnessImprovement} + 0.25 \cdot \min(\text{SeparationScore}, 2.0) + 0.2 \cdot \text{BalanceScore} + 0.25 \cdot \text{SignificanceScore}$$

## 5. Критерій поділу кластера

Кластер $C_j$ ділиться, якщо:

$$Q_{\text{split}}(C_j) \cdot \min\left(2.0, \frac{|C_j|}{50}\right) > \tau_t$$

де $\tau_t$ — адаптивний поріг:
- $\tau_t = \tau_{\text{initial}}$ для $t \leq 2$
- $\tau_t = \tau_{\text{refinement}}$ для $t > 2$

## 6. Оцінка об'єднання кластерів (зворотний прохід)

### 6.1. Збалансованість розмірів

$$\text{SizeBalance}(C_i, C_j) = \frac{\min(|C_i|, |C_j|)}{\max(|C_i|, |C_j|)}$$

### 6.2. Зміна компактності

$$\text{CompactnessChange}(C_i, C_j) = \frac{\text{AvgComp}(C_i, C_j) - \text{Comp}(C_i \cup C_j)}{\text{AvgComp}(C_i, C_j)}$$

де:

$$\text{AvgComp}(C_i, C_j) = \frac{|C_i| \cdot \text{Comp}(C_i) + |C_j| \cdot \text{Comp}(C_j)}{|C_i| + |C_j|}$$

### 6.3. Нормована відстань між центрами

$$\text{NormalizedDistance}(C_i, C_j) = \frac{\|\bm{\mu}_i - \boldsymbol{\mu}_j\|_2}{\frac{S_i + S_j}{2} + \epsilon}$$

### 6.4. Загальна оцінка об'єднання

$$Q_{\text{merge}}(C_i, C_j) = 0.3 \cdot \text{SizeBalance}(C_i, C_j) + 0.4 \cdot \text{CompactnessChange}(C_i, C_j) + 0.3 \cdot \frac{1}{\text{NormalizedDistance}(C_i, C_j) + 1}$$

## 7. Класифікація нових точок

$$\text{Score}(\mathbf{x}, C_j) = \log p(\mathbf{x} | C_j) - 0.1 \cdot \|\mathbf{x} - \boldsymbol{\mu}_j\|_2 + \log w_j$$

де:

$$\log p(\mathbf{x} | C_j) = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_j)^T \boldsymbol{\Sigma}_j^{-1} (\mathbf{x} - \boldsymbol{\mu}_j) - \frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\boldsymbol{\Sigma}_j|$$

**Правило класифікації:**

$$\hat{c}(\mathbf{x}) = \arg\max_{j} \text{Score}(\mathbf{x}, C_j)$$

## 8. Двофазний алгоритм оптимізації

### Фаза 1: Прямий прохід
1. Ініціалізація (1 кластер або KMeans)
2. Ітеративні поділи кластерів
3. Зупинка при відсутності змін

### Фаза 2: Зворотний прохід
1. Пошук пари з максимальним $Q_{\text{merge}}$
2. Об'єднання пар
3. Перевірка Silhouette Score
4. Збереження найкращої конфігурації

## 9. Метрики якості

### 9.1. Silhouette Score

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

де $a_i$ — середня відстань до свого кластера, $b_i$ — до сусіднього.

$$\text{Silhouette} = \frac{1}{n} \sum_{i=1}^n s_i$$

### 9.2. Adjusted Rand Index (ARI)

$$\text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}$$

### 9.3. Calinski-Harabasz Index

$$\text{CH} = \frac{\text{tr}(\mathbf{B})/(k-1)}{\text{tr}(\mathbf{W})/(n-k)}$$

де $\mathbf{B}$ — міжкластерна, $\mathbf{W}$ — внутрішньокластерна дисперсія.

## 10. Теоретичні властивості

### Теорема 1 (Збіжність)
Алгоритм завершується за скінченну кількість ітерацій, бо кількість кластерів обмежена.

### Теорема 2 (Монотонність)
Якість кластеризації не погіршується у зворотному проході.

### Теорема 3 (Локальна оптимальність)
Алгоритм знаходить локальний оптимум з обмеженою кількістю кластерів.

## 11. Переваги

- Автоматичне визначення $k$
- Розділення + об'єднання
- Статистичні критерії
- Адаптивні пороги
- Збереження кращого результату

## 12. Обчислювальна складність

- Прямий прохід: $O(n \cdot d \cdot k \cdot \text{max\_iter})$
- Зворотний прохід: $O(k^2 \cdot n \cdot d)$
- Загальна: $O(n \cdot d \cdot k \cdot (\text{max\_iter} + k))$
****
