import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Tworzenie figury i osi dla animacji
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Przykładowe dane do wykresu z 4 wymiarami
np.random.seed(0)
income = np.random.uniform(10000, 50000, 100)  # Przychody
profit = np.random.uniform(1000, 10000, 100)  # Zysk
user_count = np.random.uniform(100, 2000, 100)  # Liczba użytkowników
costs = np.random.uniform(2000, 20000, 100)  # Koszty (czwarty wymiar)

# Funkcja do aktualizacji wykresu w każdej klatce animacji
def update(num):
    ax.cla()  # Czyszczenie osi
    sc = ax.scatter(income, profit, user_count, c=costs, cmap='viridis', s=50, alpha=0.7)
    ax.view_init(elev=20., azim=num)  # Obracanie osi wokół osi Z
    ax.set_xlabel('Income')
    ax.set_ylabel('Profit')
    ax.set_zlabel('User Count')
    ax.set_title(f"Frame {num}")
    return sc,

# Tworzenie animacji
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

# Zapisz animację jako GIF, jeśli jesteś w środowisku, które obsługuje animacje GIF.
ani.save("3d_scatter_animation.gif", writer="pillow", fps=20)

# Wyświetlenie animacji (dla środowisk obsługujących animacje inline, jak Jupyter Notebook)
plt.show()
