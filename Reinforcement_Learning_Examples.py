import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym
from stable_baselines3.common.monitor import Monitor

# Log klasörü
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# Ortam adı
env_id = "CartPole-v1"
timesteps = 10000

# Algoritma listesi
algorithms = {
    "DQN": DQN,
    "A2C": A2C,
    "PPO": PPO
}

results = {}

# Her algoritmayı eğitme ve değerlendirme işlemi
for name, algo in algorithms.items():
    print(f"\n=== {name} eğitiliyor ===")

    # Monitor ile ortamı sarmalama işlemi
    env = gym.make("CartPole-v1")
    env = Monitor(env)

    # Model oluştuma ve eğitme işlemi
    model = algo("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(log_dir, f"{name}_tb"))
    model.learn(total_timesteps=timesteps)

    # Değerlendirme
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    results[name] = (mean_reward, std_reward)

    # Görsel test: 1 oyunluk izleme
    env = gym.make("CartPole-v1")
    obs = env.reset()
    done = False

    while not done:
        env.render()  # 👈 Bu satır önemli
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    env.close()

# Sonuçları çizme işlemi
names = list(results.keys())
means = [results[n][0] for n in names]
stds = [results[n][1] for n in names]

plt.bar(names, means, yerr=stds, capsize=5)
plt.title("Algorithm Comparison on CartPole-v1")
plt.ylabel("Mean Reward")
plt.savefig("comparison_results.png")
plt.show()

print("\nEğitim ve değerlendirme tamamlandı.")
