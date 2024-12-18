import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================================
# Definición de agentes (reemplazo de MESA Agents)
# ==========================================================================

# Estados de KPN: SC = Sensible Colonizado, RC = Resistente Colonizado, SP = Sensible Portador, RP = Resistente Portador.
KPN_STATES = ["SC", "RC", "SP", "RP"]

class KPNClone:
    """
    Agente que representa un clon de Klebsiella pneumoniae.
    Con parámetros de crecimiento y mutación variados entre clones.
    """
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.state = random.choice(KPN_STATES)
        self.growth_rate = np.random.uniform(0.5, 1.5)
        self.mutation_rate = np.random.uniform(0.01, 0.1)
        self.population = self.model.initial_kpn_population
        self.pos = None  # Posición en el grid (x, y)

    def step(self):
        # Crecimiento log-normal con leve variación
        growth_factor = np.random.lognormal(mean=0, sigma=0.4)
        new_pop = self.population * growth_factor * self.growth_rate
        self.population = max(0, int(new_pop))
        
        # Limitar la población máxima
        if self.population > 1e8:
            self.population = int(1e8)

        # Mutación de estado
        if random.random() < self.mutation_rate:
            self.state = random.choice(KPN_STATES)

class HealthcareWorker:
    """
    Agente que representa a un trabajador de la salud.
    Con movimiento aleatorio, contador de recuperación y probabilidad extra de protección.
    """
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.nivel_inmunidad = np.random.beta(a=2, b=5)
        self.prob_bioseguridad = np.random.uniform(0.7, 0.99)
        self.extra_barrier = (random.random() < 0.3)
        self.portador_kpn = False
        self.recovery_steps_remaining = 0  # Pasos restantes para recuperación
        self.pos = None

    def step(self):
        self.random_move()

        # Proceso de recuperación
        if self.portador_kpn:
            self.recovery_steps_remaining -= 1
            if self.recovery_steps_remaining <= 0:
                self.portador_kpn = False

        cellmates = self.model.get_cell_list_contents(self.pos)
        # Determinar probabilidad de transmisión
        transmission_prob = 0.15 if random.random() < self.prob_bioseguridad else 0.35
        if self.extra_barrier:
            transmission_prob *= 0.5

        for obj in cellmates:
            if obj is not self:
                if isinstance(obj, Patient) or isinstance(obj, ReusableEquipment):
                    if self.portador_kpn and random.random() < transmission_prob:
                        if isinstance(obj, Patient):
                            obj.set_infection()
                        elif isinstance(obj, ReusableEquipment):
                            obj.contaminado = True
                    if (getattr(obj, "portador_kpn", False) or getattr(obj, "contaminado", False)) \
                       and random.random() < transmission_prob:
                        self.set_infection()

    def set_infection(self):
        self.portador_kpn = True
        self.recovery_steps_remaining = 10

    def random_move(self):
        """ Mueve al trabajador aleatoriamente a una celda adyacente en el grid. """
        x, y = self.pos
        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.model.width and 0 <= ny < self.model.height:
                    candidates.append((nx, ny))
        if candidates:
            new_pos = random.choice(candidates)
            self.model.grid[x][y].remove(self)
            self.model.grid[new_pos[0]][new_pos[1]].append(self)
            self.pos = new_pos

class Patient:
    """
    Agente que representa a un paciente de UCI.
    Con movimiento aleatorio, recuperación, y uso de antibióticos.
    Variable de AISLAMIENTO: si la comorbilidad es alta, se reduce la movilidad y la prob. de contagio.
    """
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        
        raw_immune = np.random.normal(loc=0.5, scale=0.1)
        self.nivel_inmunidad = min(max(raw_immune, 0), 1)

        self.comorbilidades = np.random.poisson(lam=1.5)
        self.en_aislamiento = (self.comorbilidades > 3)

        self.recibe_antibiotico = (random.random() < 0.6)
        self.recibe_antibiotico_segunda_linea = False

        self.portador_kpn = (random.random() < 0.1)
        self.recovery_steps_remaining = 0
        self.pos = None

        if self.portador_kpn:
            self.recovery_steps_remaining = 10

    def step(self):
        # Movimiento reducido si está aislado
        if not self.en_aislamiento and random.random() < 0.02:
            self.random_move()

        # Proceso de recuperación
        if self.portador_kpn:
            self.recovery_steps_remaining -= 1
            if self.recovery_steps_remaining <= 0:
                self.portador_kpn = False

        cellmates = self.model.get_cell_list_contents(self.pos)
        base_trans_prob = 0.25
        if self.en_aislamiento:
            base_trans_prob *= 0.5

        for obj in cellmates:
            if obj is not self:
                if isinstance(obj, HealthcareWorker) or isinstance(obj, Patient):
                    if self.portador_kpn and random.random() < base_trans_prob:
                        obj.set_infection()
                    else:
                        if getattr(obj, "portador_kpn", False) and random.random() < base_trans_prob:
                            self.set_infection()

                if isinstance(obj, ReusableEquipment):
                    if self.portador_kpn and random.random() < base_trans_prob:
                        obj.contaminado = True
                    else:
                        if obj.contaminado and random.random() < base_trans_prob:
                            self.set_infection()

        # Manejo antibiótico
        if self.portador_kpn:
            is_resistant = (random.random() < 0.5)
            if is_resistant:
                if self.recibe_antibiotico:
                    self.recibe_antibiotico_segunda_linea = True
                    if random.random() < 0.1:
                        self.portador_kpn = False
                        self.recovery_steps_remaining = 0
            else:
                if self.recibe_antibiotico:
                    if random.random() < 0.3:
                        self.portador_kpn = False
                        self.recovery_steps_remaining = 0

    def set_infection(self):
        self.portador_kpn = True
        self.recovery_steps_remaining = 10

    def random_move(self):
        """ Mueve al paciente aleatoriamente a una celda adyacente. """
        x, y = self.pos
        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.model.width and 0 <= ny < self.model.height:
                    candidates.append((nx, ny))
        if candidates:
            new_pos = random.choice(candidates)
            self.model.grid[x][y].remove(self)
            self.model.grid[new_pos[0]][new_pos[1]].append(self)
            self.pos = new_pos

class ReusableEquipment:
    """
    Agente que representa un equipo médico reutilizable.
    Incluye variabilidad de 'alta tecnología' vs 'baja tecnología'.
    """
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.contaminado = (random.random() < 0.20)
        
        # Alta tecnología vs baja tecnología
        self.alta_tecnologia = (random.random() < 0.5)
        if self.alta_tecnologia:
            self.tipo_material = "alta-tec"
            self.biofilm_rate = 0.03
            self.cleanup_prob = 0.02
        else:
            self.tipo_material = "baja-tec"
            self.biofilm_rate = 0.15
            self.cleanup_prob = 0.005

        self.pos = None

    def step(self):
        # Limpieza ocasional
        if random.random() < self.cleanup_prob:
            self.contaminado = False

        if self.contaminado:
            if random.random() < self.biofilm_rate:
                pass  # Placeholder para lógica adicional si es necesario

# ==========================================================================
# Definición de la simulación sin MESA
# ==========================================================================

class ICUSimulation:
    """
    Simulación de la UCI sin MESA.
    Incluye más variabilidad y nuevas estadísticas.
    """
    def __init__(self,
                 width=10,
                 height=10,
                 initial_kpn_population=500,
                 n_patients=20,
                 n_workers=5,
                 n_equipment=5,
                 n_clones=3,
                 max_steps=50):
        
        self.width = width
        self.height = height
        self.initial_kpn_population = initial_kpn_population
        self.n_patients = n_patients
        self.n_workers = n_workers
        self.n_equipment = n_equipment
        self.n_clones = n_clones
        self.max_steps = max_steps
        
        # Grid como lista bidimensional
        self.grid = [[[] for _ in range(height)] for _ in range(width)]
        
        self.agents = []

        # Recolección de datos por paso
        self.datacollector = {
            "Total_KPN": [],
            "Pct_Pacientes_Infectados": [],
            "Pct_Trabajadores_Colonizados": [],
            "Pct_Equipos_Contaminados": [],
            "Pct_2daLinea_Pacientes": [],
            "Pct_Clones_Resistentes": []
        }

        self._create_agents()

        self.current_step = 0
        self.running = True

    def _create_agents(self):
        # Crear clones de KPN
        for i in range(self.n_clones):
            kpn_clone = KPNClone(
                unique_id=f"KPN_{i}", 
                model=self
            )
            self.agents.append(kpn_clone)
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            kpn_clone.pos = (x, y)
            self.grid[x][y].append(kpn_clone)
        
        # Crear pacientes
        for i in range(self.n_patients):
            patient = Patient(
                unique_id=f"Patient_{i}",
                model=self
            )
            self.agents.append(patient)
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            patient.pos = (x, y)
            self.grid[x][y].append(patient)
        
        # Crear trabajadores
        for i in range(self.n_workers):
            worker = HealthcareWorker(
                unique_id=f"Worker_{i}",
                model=self
            )
            self.agents.append(worker)
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            worker.pos = (x, y)
            self.grid[x][y].append(worker)
        
        # Crear equipos reutilizables
        for i in range(self.n_equipment):
            eq = ReusableEquipment(
                unique_id=f"Equipment_{i}",
                model=self
            )
            self.agents.append(eq)
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            eq.pos = (x, y)
            self.grid[x][y].append(eq)

    def get_cell_list_contents(self, pos):
        x, y = pos
        return self.grid[x][y]

    def step(self):
        self.current_step += 1

        # Ejecutar paso para cada agente
        for agent in self.agents:
            agent.step()

        # Recolectar datos
        self.collect_data()

        # Verificar criterio de parada
        if self.current_step >= self.max_steps:
            self.running = False

    def collect_data(self):
        self.datacollector["Total_KPN"].append(self.compute_total_kpn())
        self.datacollector["Pct_Pacientes_Infectados"].append(self.compute_pct_patients_infected())
        self.datacollector["Pct_Trabajadores_Colonizados"].append(self.compute_pct_workers_colonized())
        self.datacollector["Pct_Equipos_Contaminados"].append(self.compute_pct_equipment_contaminated())
        self.datacollector["Pct_2daLinea_Pacientes"].append(self.compute_pct_second_line_patients())
        self.datacollector["Pct_Clones_Resistentes"].append(self.compute_pct_resistant_clones())

    # ==========================================================================
    # Cálculos estadísticos
    # ==========================================================================
    def compute_total_kpn(self):
        total_kpn = sum(agent.population for agent in self.agents if isinstance(agent, KPNClone))
        return total_kpn

    def compute_pct_patients_infected(self):
        patients = [a for a in self.agents if isinstance(a, Patient)]
        if not patients:
            return 0
        infected = sum(1 for p in patients if p.portador_kpn)
        return infected / len(patients)

    def compute_pct_workers_colonized(self):
        workers = [a for a in self.agents if isinstance(a, HealthcareWorker)]
        if not workers:
            return 0
        colonized = sum(1 for w in workers if w.portador_kpn)
        return colonized / len(workers)

    def compute_pct_equipment_contaminated(self):
        eq_list = [a for a in self.agents if isinstance(a, ReusableEquipment)]
        if not eq_list:
            return 0
        contaminated = sum(1 for eq in eq_list if eq.contaminado)
        return contaminated / len(eq_list)

    def compute_pct_second_line_patients(self):
        patients = [a for a in self.agents if isinstance(a, Patient)]
        if not patients:
            return 0
        second_line = sum(1 for p in patients if p.recibe_antibiotico_segunda_linea)
        return second_line / len(patients)

    def compute_pct_resistant_clones(self):
        clones = [a for a in self.agents if isinstance(a, KPNClone)]
        if not clones:
            return 0
        resistant = sum(1 for c in clones if c.state in ["RC", "RP"])
        return resistant / len(clones)

# ==========================================================================
# Funciones para correr la simulación con múltiples réplicas y generar más gráficas
# ==========================================================================

def single_run_simulation(params):
    """
    Ejecuta la simulación una sola vez con los parámetros dados en 'params'.
    Retorna el diccionario de datos recolectados (listas) y el número de pasos ejecutados.
    """
    model = ICUSimulation(**params)
    while model.running:
        model.step()
    return model.datacollector, model.current_step

def multi_run_simulation(n_runs=10):
    """
    Corre la simulación n_runs veces, guarda los datos en listas, y produce
    gráficas agregadas (promedio y std) por paso.
    """
    # Parámetros base
    params = {
        'width': 20,
        'height': 20,
        'initial_kpn_population': 500,
        'n_patients': 30,
        'n_workers': 8,
        'n_equipment': 10,
        'n_clones': 5,
        'max_steps': 40
    }

    runs_data = []
    max_steps_encountered = 0

    for run_idx in range(n_runs):
        data_collector, steps_done = single_run_simulation(params)
        runs_data.append(data_collector)
        if steps_done > max_steps_encountered:
            max_steps_encountered = steps_done

    var_names = [
        "Total_KPN",
        "Pct_Pacientes_Infectados",
        "Pct_Trabajadores_Colonizados",
        "Pct_Equipos_Contaminados",
        "Pct_2daLinea_Pacientes",
        "Pct_Clones_Resistentes"
    ]

    # Estructura: results[var_name][step][run]
    results = {var: np.zeros((max_steps_encountered, n_runs)) for var in var_names}

    for run_idx, data_collector in enumerate(runs_data):
        for var in var_names:
            values = data_collector[var]
            for step_idx, val in enumerate(values):
                results[var][step_idx, run_idx] = val

    # Calcular media y desviación estándar
    steps = range(max_steps_encountered)

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs = axs.ravel()

    for i, var in enumerate(var_names):
        mean_vals = np.mean(results[var], axis=1)
        std_vals  = np.std(results[var], axis=1)
        ax = axs[i]
        ax.plot(steps, mean_vals, label=f"{var} (media)")
        ax.fill_between(steps,
                        mean_vals - std_vals,
                        mean_vals + std_vals,
                        alpha=0.3, label=f"{var} (±1σ)")
        ax.set_title(var)
        ax.set_xlabel("Paso")
        if var == "Total_KPN":
            ax.set_ylabel("Bacterias (unidades abstractas)")
        else:
            ax.set_ylabel("Proporción")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Boxplot final en el último paso de cada réplica para cada variable
    fig2, axs2 = plt.subplots(1, 3, figsize=(15,5))

    final_metrics = ["Total_KPN", "Pct_Pacientes_Infectados", "Pct_Trabajadores_Colonizados"]
    for i, var in enumerate(final_metrics):
        last_vals = [runs_data[run_idx][var][-1] for run_idx in range(n_runs)]
        axs2[i].boxplot(last_vals, vert=True)
        axs2[i].set_title(f"Distribución final {var} (último paso)")
        axs2[i].set_ylabel(var)

    plt.tight_layout()
    plt.show()

def run_simulation_multiple_replicas():
    """
    Ejecuta la simulación en modo multi-run con 50 réplicas y produce gráficas agregadas.
    """
    multi_run_simulation(n_runs=50)

if __name__ == "__main__":
    # Ejecutar la simulación con múltiples réplicas
    run_simulation_multiple_replicas()
