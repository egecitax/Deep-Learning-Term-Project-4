# YZM304 4. Ödev – Derin Pekiştirmeli Öğrenme Karşılaştırması

## Giriş

Pekiştirmeli öğrenme, bir ajan ile ortam arasındaki etkileşime dayalı bir öğrenme paradigmasıdır. Bu projede, OpenAI Gym ortamlarından biri olan `CartPole-v1` üzerinde üç farklı derin pekiştirmeli öğrenme algoritması karşılaştırılmıştır: **DQN**, **A2C** ve **PPO**. Amaç, bu algoritmaların belirli bir görevdeki ortalama başarılarını karşılaştırmak ve hangisinin daha verimli öğrendiğini analiz etmektir.

---

## Yöntem

### Ortam

- **Gym Ortamı:** CartPole-v1
- **Gözlem:** 4 boyutlu sürekli değerler (konum, hız, açı, açısal hız)
- **Aksiyon:** 2 (sola ya da sağa hareket)

### Kullanılan Algoritmalar

- **DQN (Deep Q-Network):** Q-learning tabanlı, ayrık aksiyon uzayı için uygundur.
- **A2C (Advantage Actor-Critic):** Policy Gradient + Value function yapısını birlikte kullanır.
- **PPO (Proximal Policy Optimization):** Policy Gradient yöntemlerini daha stabil hale getirir.

### Eğitim Yapılandırması

- **Zaman Adımı:** Her model için 10.000 `timesteps`
- **Policy:** `MlpPolicy`
- **Loglama:** `Monitor` + `TensorBoard` ile loglar kaydedildi.
- **Değerlendirme:** Her model, eğitim sonrası 10 epizotta `evaluate_policy` ile test edildi.

---

## Sonuçlar

Aşağıda algoritmaların eğitim sonrası ortalama ödüllerine göre karşılaştırılması gösterilmektedir:

![Karşılaştırma Grafiği](comparison_results.png)

| Algoritma | Ortalama Ödül | Standart Sapma |
|-----------|----------------|----------------|
| **DQN**   | Çok düşük      | ± küçük        |
| **A2C**   | Orta düzey     | ± yüksek       |
| **PPO**   | Maksimuma yakın| ± düşük        |

- **PPO** algoritması, CartPole ortamında maksimum ödüle ulaşarak en yüksek performansı göstermiştir.
- **DQN**, epsilon-greedy keşif stratejisiyle istikrarsız bir öğrenme süreci geçirmiştir.
- **A2C**, daha stabil ancak PPO kadar etkili değildir.

---

## Tartışma

- PPO'nun avantajı, büyük aksiyon alanlarında bile kararlı şekilde öğrenme gerçekleştirmesidir.
- DQN gibi değer temelli yöntemler, ayrık aksiyon uzaylarında çalışmasına rağmen derinlik eksikliği nedeniyle başarısız olabilmektedir.
- A2C ise Actor-Critic mimarisi sayesinde denge kurar fakat parametre ayarlarına duyarlıdır.
- Eğitim sırasında TensorBoard ile **reward**, **loss**, **entropy** gibi metrikler gözlemlenmiştir.

---

## Referanslar

1. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.*
2. Mnih et al. (2015). *Human-level control through deep reinforcement learning.*
3. OpenAI Gym: https://www.gymlibrary.dev/
4. Stable Baselines3: https://github.com/DLR-RM/stable-baselines3

---

## Kurulum

```bash
pip install -r requirements.txt
