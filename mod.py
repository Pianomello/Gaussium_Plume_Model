import numpy as np
import matplotlib.pyplot as plt

# Set up stability parameters
a = { 'A': 213, 'B': 156, 'C': 104, 'D': 68, 'E': 50.5, 'F': 34 }
c_0 = { 'A': 440.8, 'B': 106.6, 'C': 61.0, 'D': 33.2, 'E': 22.8, 'F': 14.35 }
d_0 = { 'A': 1.941, 'B': 1.149, 'C': 0.911, 'D': 0.725, 'E': 0.678, 'F': 0.740 }
f_0 = { 'A': 9.27, 'B': 3.3, 'C': 0, 'D': -1.7, 'E': -1.3, 'F': -0.35 }
c_1 = { 'A': 459.7, 'B': 108.2, 'C': 61.0, 'D': 44.5, 'E': 55.4, 'F': 62.6 }
d_1 = { 'A': 2.094, 'B': 1.098, 'C': 0.911, 'D': 0.516, 'E': 0.305, 'F': 0.180 }
f_1 = { 'A': -9.6, 'B': 2.0, 'C': 0, 'D': -13.0, 'E': -34.0, 'F': -48.6 }
p = { 'A': 0.15, 'B': 0.15, 'C': 0.20, 'D': 0.25, 'E': 0.40, 'F': 0.60 }

class GPM:
    def __init__(self, Q, stability):
        self.Q = Q
        self.a = a[stability]
        self.c_0 = c_0[stability]
        self.d_0 = d_0[stability]
        self.f_0 = f_0[stability]
        self.c_1 = c_1[stability]
        self.d_1 = d_1[stability]
        self.f_1 = f_1[stability]
        self.p = p[stability]

    def stack(self, h, T_a, T_s, v_s, r, u_h):
        self.F = 9.8 * r**2 * v_s * (1 - float(T_a) / T_s)  
        
        # **Limit excessive plume rise to keep the peak concentration closer**
        if self.F >= 55:
            x_f = 100 * self.F ** 0.4  
        else:
            x_f = 40 * self.F ** 0.625  
        
        self.dH = min(50, 1.4 * self.F ** (1.0 / 3) * x_f ** (2.0 / 3) / u_h)  
        self.H = h + self.dH  
        self.u_H = u_h * (self.H / 10.0) ** self.p  
        self.T_a = float(T_a)

        print(f"Optimized H = {self.H:.2f} m")
        print(f"Optimized u_H = {self.u_H:.2f} m/s")

    def smooth_sigma_z(self, x_km, k=10):
        """ Smooth transition between σ_z equations """
        S_x = 1 / (1 + np.exp(-k * (x_km - 1)))  
        s_z1 = self.c_0 * x_km ** self.d_0 + self.f_0  
        s_z2 = self.c_1 * x_km ** self.d_1 + self.f_1  
        return (1 - S_x) * s_z1 + S_x * s_z2  

    def conc(self, x, y):
        x_km = x / 1000.0  
        s_y = self.a * x_km ** 0.894  
        s_z = self.smooth_sigma_z(x_km)  
        return self.Q / (np.pi * self.u_H * s_y * s_z * 1000) * \
               np.exp(-1 * (self.H ** 2 / (2 * s_z ** 2))) * \
               np.exp(-1 * (y ** 2 / (2 * s_y ** 2)))
               
    def conc_t(self, x, y):
        x_km = x / 1000.0
        s_y = self.a * x_km ** 0.894
        s_z = self.smooth_sigma_z(x_km)

        # Modify stack height `H` based on ambient temperature
        self.H = self.H + 0.1 * (self.T_a - 300)  # Adjust rise height

        # Modify s_z based on temperature (higher temperature causes more dispersion)
        s_z *= (1 + (self.T_a - 300) / 100)  # Adjust dispersion based on temperature difference

        # Calculate concentration using the adjusted values
        return self.Q / (np.pi * s_y * s_z) * \
               np.exp(-1 * (self.H ** 2 / (2 * s_z ** 2))) * \
               np.exp(-1 * (y ** 2 / (2 * s_y ** 2)))

# Create grid (focus on 0-10 km for better resolution)
x = np.linspace(0, 20000, 1000)  
y = np.linspace(-5000, 5000, 1000)  
X, Y = np.meshgrid(x, y)

# Initialize and optimize parameters
model = GPM(1.0944e9, 'C')  
model.stack(h=220, T_a=307.34, T_s=413.15, r=2, v_s=28.7, u_h=4.68)  

# Compute concentration values
Z = model.conc(X, Y)

# Plot downwind concentration profile
x_range = np.linspace(0, 30000, 500)  
C_values = model.conc(x_range, 0)  

plt.plot(x_range / 1000, C_values, label="Optimized Concentration")
plt.xlabel("Downwind Distance (km)")
plt.ylabel("Concentration (µg/m³)")
plt.title(f"Peak Concentration at {model.H:.2f} m Stack Height")
plt.legend()
plt.grid()
plt.savefig("optimized_gaussian_plume.png")
plt.show()

# Check concentration at key distances
print("Concentration at 10 km:", model.conc(10000, 0))
print("Concentration at 15 km:", model.conc(15000, 0))
print("Concentration at 20 km:", model.conc(20000, 0))



# Plot a contour map
fig1 = plt.figure()
plt.contourf(X, Y, Z, levels=200, cmap="plasma")
colorbar = plt.colorbar()
colorbar.set_label("SO$_{2}$ ($\mu g / m^3$)")
contours = plt.contour(X, Y, Z,
                       levels=[1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50],
                       colors="red", linewidths=0.5, linestyles="dotted")
plt.xlabel("$x$ (m)")
plt.ylabel("$y$ (m)")
plt.title("Concentrations of SO$_{2}$")
plt.clabel(contours, inline=True, colors="black", fmt="%d", fontsize=8)
filename = f"H_stubble_{model.H:.2f}.png"
plt.savefig(filename)
plt.clf()
print(f"Filename: {filename}")



# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

# Labels and title
ax.set_xlabel('Downwind Distance (m)')
ax.set_ylabel('Crosswind Distance (m)')
ax.set_zlabel('Concentration (µg/m³)')
ax.set_title('3D Plot of Concentration vs Downwind Distance')

# Show plot
plt.show()




# Variation of downwind concentration with windspeed for a particular ambient temperature
fig3 = plt.subplots()

model.stack(h=220, T_a=307.34, T_s=413.15, r=2, v_s=28.7, u_h=3.889)
plt.plot(x, model.conc(x, x * 0), label="$u_h = 3.889$ m/s")
print("Concentration downwind at x = 5 km:", model.conc(5000, 0))

model.stack(h=220, T_a=307.34, T_s=413.15, r=2, v_s=28.7, u_h=8.527)
plt.plot(x, model.conc(x, x * 0), label="$u_h = 8.527$ m/s")
print("Concentration downwind at x = 5 km:", model.conc(5000, 0))

model.stack(h=220, T_a=307.34, T_s=413.15, r=2, v_s=28.7, u_h=4.251)
plt.plot(x, model.conc(x, x * 0), label="$u_h = 4.251$ m/s")
print("Concentration downwind at x = 5 km:", model.conc(5000, 0))


model.stack(h=220, T_a=307.34, T_s=413.15, r=2, v_s=28.7, u_h=10.0)
plt.plot(x, model.conc(x, x * 0), label="$u_h = 10$ m/s")
print("Concentration downwind at x = 5 km:", model.conc(5000, 0))

# Set plot labels and title
plt.xlabel("Distance downwind (m)")
plt.ylabel("Concentration (µg/m³)")
plt.title("Effect of wind speed on downwind concentration")
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig("windspeed_variation.png")
plt.show()




# Variation of downwind concentration with average ambient temperature of all the 4 seasons
fig4 = plt.subplots()

# Summer temperature
model.stack(h=220, T_a=312.28, T_s=413.15, r=2, v_s=28.7, u_h=4.68)
plt.plot(x, model.conc_t(x, x * 0), label="$Summer = 312.28$ K")
print("Concentration downwind at x = 5 km:", model.conc_t(5000, 0))

# Monsoon temperature
model.stack(h=220, T_a=307.97, T_s=413.15, r=2, v_s=28.7, u_h=4.68)
plt.plot(x, model.conc_t(x, x * 0), label="$Monsoon = 307.97$ K")
print("Concentration downwind at x = 5 km:", model.conc_t(5000, 0))

# Autumn temperature
model.stack(h=220, T_a=304.5, T_s=413.15, r=2, v_s=28.7, u_h=4.68)
plt.plot(x, model.conc_t(x, x * 0), label="$Autumn = 304.5$ K")
print("Concentration downwind at x = 5 km:", model.conc_t(5000, 0))

# Winter temperature
model.stack(h=220, T_a=304.58, T_s=413.15, r=2, v_s=28.7, u_h=4.68)
plt.plot(x, model.conc_t(x, x * 0), label="$Winter = 304.58$ K")
print("Concentration downwind at x = 5 km:", model.conc_t(5000, 0))

# Set plot labels and title
plt.xlabel("Distance downwind (m)")
plt.ylabel("Concentration (µg/m³)")
plt.title("Downwind Concentration vs Ambient Temperature")
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig("Variation_with_T_a.png")
plt.show()























