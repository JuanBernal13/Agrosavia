import numpy as np
from agents import KPNClone, Patient, HealthcareWorker, ReusableEquipment
import random

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

        print("[INIT] Creando simulación UCI.")
        
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
        print("[INIT] Creando agentes en la simulación.")
        for i in range(self.n_clones):
            kpn_clone = KPNClone(
                unique_id=f"KPN_{i}", 
                model=self
            )
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            kpn_clone.pos = (x, y)
            self.grid[x][y].append(kpn_clone)
            self.agents.append(kpn_clone)
        
        for i in range(self.n_patients):
            patient = Patient(
                unique_id=f"Patient_{i}",
                model=self
            )
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            patient.pos = (x, y)
            self.grid[x][y].append(patient)
            self.agents.append(patient)
        
        for i in range(self.n_workers):
            worker = HealthcareWorker(
                unique_id=f"Worker_{i}",
                model=self
            )
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            worker.pos = (x, y)
            self.grid[x][y].append(worker)
            self.agents.append(worker)
        
        for i in range(self.n_equipment):
            eq = ReusableEquipment(
                unique_id=f"Equipment_{i}",
                model=self
            )
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            eq.pos = (x, y)
            self.grid[x][y].append(eq)
            self.agents.append(eq)

    def get_cell_list_contents(self, pos):
        x, y = pos
        return self.grid[x][y]

    def step(self):
        self.current_step += 1
        print(f"[MODEL STEP] Paso {self.current_step} de {self.max_steps}")
        for agent in self.agents:
            agent.step()

        self.collect_data()

        if self.current_step >= self.max_steps:
            print("[MODEL] Simulación terminada.")
            self.running = False

    def collect_data(self):
        print("[DATA] Recolectando datos del paso.")
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

    def compute_total_kpn(self):
        total = sum(agent.population for agent in self.agents if hasattr(agent, "population"))
        return total

    def compute_pct_patients_infected(self):
        patients = [a for a in self.agents if hasattr(a, "portador_kpn") and isinstance(a, Patient)]
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
