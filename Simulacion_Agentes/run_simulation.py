import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# Definición de agentes
# =====================================================================

KPN_STATES = ["SC", "RC", "SP", "RP"]

class KPNClone:
    """
    Agente que representa un clon de Klebsiella pneumoniae.
    Ahora puede también "diseminar" parte de su población a celdas vecinas con menor carga bacteriana.
    """
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.state = random.choice(KPN_STATES)
        self.growth_rate = np.random.uniform(0.5, 1.5)
        self.mutation_rate = np.random.uniform(0.01, 0.1)
        self.resistencia_acumulada = np.random.uniform(0, 0.3)
        self.tasa_mortalidad = np.random.uniform(0.001, 0.01)
        self.population = self.model.initial_kpn_population
        self.pos = None

    def step(self):
        # Crecimiento con ruido lognormal
        growth_factor = np.random.lognormal(mean=0, sigma=0.4)
        new_pop = self.population * growth_factor * self.growth_rate
        # Mortalidad natural
        new_pop = new_pop * (1 - self.tasa_mortalidad)
        self.population = max(0, int(new_pop))

        # Limitar la población máxima
        if self.population > 1e8:
            self.population = int(1e8)

        # Mutación de estado con sesgo
        if random.random() < (self.mutation_rate + self.resistencia_acumulada):
            if self.state in ["SC", "SP"]:
                if random.random() < 0.5 + self.resistencia_acumulada:
                    self.state = random.choice(["RC", "RP"])
                else:
                    self.state = random.choice(["SC", "SP"])
            else:
                if random.random() < 0.8:
                    self.state = random.choice(["RC", "RP"])
                else:
                    self.state = random.choice(["SC", "SP"])

        # Diseminación: si la población es alta, intento "migrar" algo de población a una celda vecina.
        if self.population > 1000:
            self.diseminar()

    def diseminar(self):
        x, y = self.pos
        candidates = []
        # Buscamos celdas con baja carga KPN
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.model.width and 0 <= ny < self.model.height:
                    # Ver si hay clones allí
                    cell_clones = [a for a in self.model.grid[nx][ny] if isinstance(a, KPNClone)]
                    total_pop = sum(c.population for c in cell_clones)
                    if total_pop < self.population * 0.5:
                        candidates.append((nx, ny, total_pop))

        if candidates:
            # Escoger la celda con menor población (intentar colonizarla)
            candidates.sort(key=lambda c: c[2])  # ordenamos por total_pop asc
            target = candidates[0]
            nx, ny = target[0], target[1]

            # Crear un nuevo clon en la celda vecina (o incrementar si hay uno)
            cell_clones = [a for a in self.model.grid[nx][ny] if isinstance(a, KPNClone)]
            trans_amount = int(self.population * 0.1)  # 10% de la población migra
            if not cell_clones:
                # Clonar el agente
                new_clone = KPNClone(unique_id=f"{self.unique_id}_migrate", model=self.model)
                new_clone.state = self.state
                new_clone.population = trans_amount
                new_clone.pos = (nx, ny)
                self.model.agents.append(new_clone)
                self.model.grid[nx][ny].append(new_clone)
            else:
                # Incrementar población de uno de los clones existentes
                chosen_clone = random.choice(cell_clones)
                chosen_clone.population += trans_amount

            self.population -= trans_amount


class HealthcareWorker:
    """
    Trabajador de la salud con mayor toma de decisiones:
    - Si no está infectado, puede intentar ir hacia un equipo contaminado para "ayudar" a limpiarlo.
    - Si está infectado, puede intentar moverse hacia celdas con menos agentes para reducir contagios.
    """
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.nivel_inmunidad = np.random.beta(a=2, b=5)
        self.prob_bioseguridad = np.random.uniform(0.7, 0.99)
        self.extra_barrier = (random.random() < 0.3)
        self.portador_kpn = False
        self.recovery_steps_remaining = 0
        self.reinfection_resistance = np.random.uniform(0, 0.2)
        self.pos = None

    def step(self):
        # Decisiones de movimiento:
        if self.portador_kpn:
            # Infectado: intentar celdas menos pobladas (menos contagio)
            self.move_to_less_crowded()
        else:
            # Sano: intentar ir hacia un equipo contaminado para poder "ayudar" a limpiarlo
            # con cierta probabilidad
            if random.random() < 0.3:
                self.move_towards_contaminated_equipment()
            else:
                self.random_move()

        # Proceso de recuperación
        if self.portador_kpn:
            self.recovery_steps_remaining -= 1
            if self.recovery_steps_remaining <= 0:
                self.portador_kpn = False
                self.reinfection_resistance += 0.05

        # Interacciones en la celda
        cellmates = self.model.get_cell_list_contents(self.pos)
        base_transmission_prob = 0.15 if random.random() < self.prob_bioseguridad else 0.35
        if self.extra_barrier:
            base_transmission_prob *= 0.5

        for obj in cellmates:
            if obj is not self:
                if isinstance(obj, Patient) or isinstance(obj, ReusableEquipment):
                    if self.portador_kpn and random.random() < base_transmission_prob:
                        if isinstance(obj, Patient):
                            obj.set_infection()
                        elif isinstance(obj, ReusableEquipment):
                            obj.contaminado = True
                    if (getattr(obj, "portador_kpn", False) or getattr(obj, "contaminado", False)) \
                       and random.random() < (base_transmission_prob * (1 - self.reinfection_resistance)):
                        self.set_infection()

    def set_infection(self):
        self.portador_kpn = True
        self.recovery_steps_remaining = random.randint(5, 15)

    def move_towards_contaminated_equipment(self):
        # Buscar la celda vecina que tenga mayor probabilidad de contener equipo contaminado
        x, y = self.pos
        best_cell = None
        best_score = -1
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.model.width and 0 <= ny < self.model.height and not (dx ==0 and dy ==0):
                    eq = [a for a in self.model.grid[nx][ny] if isinstance(a, ReusableEquipment) and a.contaminado]
                    score = len(eq)
                    if score > best_score:
                        best_score = score
                        best_cell = (nx, ny)

        if best_cell is not None and best_score > 0:
            self.move_to(best_cell)
        else:
            self.random_move()

    def move_to_less_crowded(self):
        # Infectado: buscar celda vecina con menos agentes para reducir contactos
        x, y = self.pos
        best_cell = None
        best_count = 9999
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0:
                    continue
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.model.width and 0 <= ny < self.model.height:
                    count_agents = len(self.model.grid[nx][ny])
                    if count_agents < best_count:
                        best_count = count_agents
                        best_cell = (nx, ny)
        if best_cell is not None:
            self.move_to(best_cell)
        else:
            self.random_move()

    def move_to(self, new_pos):
        x,y = self.pos
        self.model.grid[x][y].remove(self)
        self.model.grid[new_pos[0]][new_pos[1]].append(self)
        self.pos = new_pos

    def random_move(self):
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
            self.move_to(new_pos)


class Patient:
    """
    Paciente con más lógica de movimiento:
    - Si sano, trata de alejarse de equipos contaminados y agentes infectados.
    - Si infectado, puede intentar moverse hacia equipos de alta tecnología, 
      pues podrían asociarse a menor acumulación bacteriana.
    - Si muy inmunodeprimido, moverse a celdas menos congestionadas.
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
        self.terapia_combinada = False

        self.portador_kpn = (random.random() < 0.1)
        self.recovery_steps_remaining = 0
        self.pos = None

        if self.portador_kpn:
            self.recovery_steps_remaining = 10

    def step(self):
        # Decisiones de movimiento
        if self.portador_kpn:
            # Infectado: buscar celdas con equipos de alta tecnología (asumiendo menor biofilm)
            if random.random() < 0.3:
                self.move_towards_hitech_equipment()
            else:
                self.random_move()
        else:
            # Sano: alejarse de celdas con equipo contaminado o personas infectadas
            if random.random() < 0.3:
                self.move_away_from_contamination()
            else:
                self.random_move()

        # Proceso de recuperación
        if self.portador_kpn:
            self.recovery_steps_remaining -= 1
            if self.recovery_steps_remaining <= 0:
                self.portador_kpn = False

        # Interacción en la celda
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
                    self.terapia_combinada = True
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

    def move_away_from_contamination(self):
        # Alejarse de celdas con equipo contaminado o personas infectadas
        x,y = self.pos
        best_cell = None
        best_risk = 9999
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0:
                    continue
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.model.width and 0 <= ny < self.model.height:
                    cellmates = self.model.grid[nx][ny]
                    risk = sum(1 for c in cellmates if (isinstance(c, Patient) or isinstance(c, HealthcareWorker)) and c.portador_kpn)
                    risk += sum(2 for c in cellmates if isinstance(c, ReusableEquipment) and c.contaminado)
                    if risk < best_risk:
                        best_risk = risk
                        best_cell = (nx, ny)
        if best_cell is not None:
            self.move_to(best_cell)
        else:
            self.random_move()

    def move_towards_hitech_equipment(self):
        # Infectado: buscar celdas con equipo de alta tecnología, quizá reduce la resistencia
        x,y = self.pos
        hitech_candidates = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0:
                    continue
                nx, ny = x+dx, y+dy
                if 0<=nx<self.model.width and 0<=ny<self.model.height:
                    eq = [a for a in self.model.grid[nx][ny] if isinstance(a, ReusableEquipment) and a.alta_tecnologia]
                    if eq:
                        hitech_candidates.append((nx, ny))
        if hitech_candidates:
            new_pos = random.choice(hitech_candidates)
            self.move_to(new_pos)
        else:
            self.random_move()

    def move_to(self, new_pos):
        x,y = self.pos
        self.model.grid[x][y].remove(self)
        self.model.grid[new_pos[0]][new_pos[1]].append(self)
        self.pos = new_pos

    def random_move(self):
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
            self.move_to(new_pos)


class ReusableEquipment:
    """
    Equipo reutilizable:
    - Si un trabajador llega a su celda, incrementa la probabilidad de ser limpiado (simulado indirectamente).
    """
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.contaminado = (random.random() < 0.20)
        
        self.alta_tecnologia = (random.random() < 0.5)
        if self.alta_tecnologia:
            self.tipo_material = "alta-tec"
            self.biofilm_rate = 0.03
            self.cleanup_prob = 0.02
        else:
            self.tipo_material = "baja-tec"
            self.biofilm_rate = 0.15
            self.cleanup_prob = 0.005

        self.tiempo_contaminado = 0
        self.pos = None

    def step(self):
        # Si hay un trabajador en la misma celda, incrementar un poco la prob. de limpieza
        cellmates = self.model.get_cell_list_contents(self.pos)
        if any(isinstance(a, HealthcareWorker) for a in cellmates):
            effective_cleanup = self.cleanup_prob + 0.05
        else:
            effective_cleanup = self.cleanup_prob

        if self.contaminado and random.random() < effective_cleanup:
            self.contaminado = False
            self.tiempo_contaminado = 0

        if self.contaminado:
            self.tiempo_contaminado += 1
            # Incrementa el biofilm_rate con el tiempo
            self.biofilm_rate = min(self.biofilm_rate + 0.001, 0.3)


# =====================================================================
# Definición de la simulación
# =====================================================================

class ICUSimulation:
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
        
        # Grid
        self.grid = [[[] for _ in range(height)] for _ in range(width)]
        
        self.agents = []

        # Recolección de datos
        self.datacollector = {
            "Total_KPN": [],
            "Pct_Pacientes_Infectados": [],
            "Pct_Trabajadores_Colonizados": [],
            "Pct_Equipos_Contaminados": [],
            "Pct_2daLinea_Pacientes": [],
            "Pct_Clones_Resistentes": [],
            "Pct_SC_Clones": [],
            "Pct_RC_Clones": [],
            "Pct_SP_Clones": [],
            "Pct_RP_Clones": []
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
        
        # Crear equipos
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

        for agent in self.agents:
            agent.step()

        self.collect_data()

        if self.current_step >= self.max_steps:
            self.running = False

    def collect_data(self):
        self.datacollector["Total_KPN"].append(self.compute_total_kpn())
        self.datacollector["Pct_Pacientes_Infectados"].append(self.compute_pct_patients_infected())
        self.datacollector["Pct_Trabajadores_Colonizados"].append(self.compute_pct_workers_colonized())
        self.datacollector["Pct_Equipos_Contaminados"].append(self.compute_pct_equipment_contaminated())
        self.datacollector["Pct_2daLinea_Pacientes"].append(self.compute_pct_second_line_patients())
        self.datacollector["Pct_Clones_Resistentes"].append(self.compute_pct_resistant_clones())

        sc, rc, sp, rp = self.compute_clone_state_distribution()
        self.datacollector["Pct_SC_Clones"].append(sc)
        self.datacollector["Pct_RC_Clones"].append(rc)
        self.datacollector["Pct_SP_Clones"].append(sp)
        self.datacollector["Pct_RP_Clones"].append(rp)

    # =====================================================================
    # Cálculos estadísticos
    # =====================================================================
    def compute_total_kpn(self):
        return sum(agent.population for agent in self.agents if isinstance(agent, KPNClone))

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
        second_line = sum(1 for p in patients if p.recibe_antibiotico_segunda_linea or p.terapia_combinada)
        return second_line / len(patients)

    def compute_pct_resistant_clones(self):
        clones = [a for a in self.agents if isinstance(a, KPNClone)]
        if not clones:
            return 0
        resistant = sum(1 for c in clones if c.state in ["RC", "RP"])
        return resistant / len(clones)

    def compute_clone_state_distribution(self):
        clones = [a for a in self.agents if isinstance(a, KPNClone)]
        if not clones:
            return (0, 0, 0, 0)
        total = len(clones)
        sc = sum(1 for c in clones if c.state == "SC") / total
        rc = sum(1 for c in clones if c.state == "RC") / total
        sp = sum(1 for c in clones if c.state == "SP") / total
        rp = sum(1 for c in clones if c.state == "RP") / total
        return (sc, rc, sp, rp)


# =====================================================================
# Funciones de simulación
# =====================================================================

def single_run_simulation(params):
    model = ICUSimulation(**params)
    while model.running:
        model.step()
    return model.datacollector, model.current_step, model

def multi_run_simulation(n_runs=10):
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
        data_collector, steps_done, _ = single_run_simulation(params)
        runs_data.append(data_collector)
        if steps_done > max_steps_encountered:
            max_steps_encountered = steps_done

    var_names = [
        "Total_KPN",
        "Pct_Pacientes_Infectados",
        "Pct_Trabajadores_Colonizados",
        "Pct_Equipos_Contaminados",
        "Pct_2daLinea_Pacientes",
        "Pct_Clones_Resistentes",
        "Pct_SC_Clones",
        "Pct_RC_Clones",
        "Pct_SP_Clones",
        "Pct_RP_Clones"
    ]

    results = {var: np.zeros((max_steps_encountered, n_runs)) for var in var_names}

    for run_idx, data_collector in enumerate(runs_data):
        for var in var_names:
            values = data_collector[var]
            for step_idx, val in enumerate(values):
                results[var][step_idx, run_idx] = val

    steps = range(max_steps_encountered)

    fig, axs = plt.subplots(5, 2, figsize=(12, 20))
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
            ax.set_ylabel("Bacterias (u.a.)")
        else:
            ax.set_ylabel("Proporción")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Boxplot final en el último paso
    fig2, axs2 = plt.subplots(1, 3, figsize=(15,5))
    final_metrics = ["Total_KPN", "Pct_Pacientes_Infectados", "Pct_Trabajadores_Colonizados"]
    for i, var in enumerate(final_metrics):
        last_vals = [runs_data[run_idx][var][-1] for run_idx in range(n_runs)]
        axs2[i].boxplot(last_vals, vert=True)
        axs2[i].set_title(f"Distribución final {var} (último paso)")
        axs2[i].set_ylabel(var)

    plt.tight_layout()
    plt.show()

    # Histograma final de distribución de estados de clones en el último paso
    final_run_data = runs_data[0]
    sc_last = final_run_data["Pct_SC_Clones"][-1]
    rc_last = final_run_data["Pct_RC_Clones"][-1]
    sp_last = final_run_data["Pct_SP_Clones"][-1]
    rp_last = final_run_data["Pct_RP_Clones"][-1]

    labels = ["SC", "RC", "SP", "RP"]
    dist = [sc_last, rc_last, sp_last, rp_last]

    plt.figure(figsize=(6,4))
    plt.bar(labels, dist)
    plt.title("Distribución final de estados KPN (un run)")
    plt.ylabel("Proporción")
    plt.show()

def run_simulation_multiple_replicas():
    multi_run_simulation(n_runs=50)


if __name__ == "__main__":
    run_simulation_multiple_replicas()
