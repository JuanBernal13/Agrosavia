import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Datos de ejemplo
np.random.seed(0)
datos = np.random.rand(10, 12)
correlacion = np.corrcoef(datos)

# Crear el heatmap
sns.heatmap(correlacion, annot=True, cmap='coolwarm')
plt.title('Mapa de Calor de Correlación')
plt.show()
