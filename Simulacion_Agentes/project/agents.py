import random
import numpy as np

KPN_STATES = ["SC", "RC", "SP", "RP"]

class KPNClone:
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
        print(f"[INIT] KPNClone {self.unique_id} creado en estado {self.state} con población {self.population}")

    def step(self):
        print(f"[STEP] KPNClone {self.unique_id} en posición {self.pos}, estado {self.state}, población {self.population}")
        growth_factor = np.random.lognormal(mean=0, sigma=0.4)
        new_pop = self.population * growth_factor * self.growth_rate
        new_pop = new_pop * (1 - self.tasa_mortalidad)
        self.population = max(0, int(new_pop))

        if self.population > 1e8:
            self.population = int(1e8)

        # Mutación
        if random.random() < (self.mutation_rate + self.resistencia_acumulada):
            old_state = self.state
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
            print(f"[MUTATION] KPNClone {self.unique_id} muta de {old_state} a {self.state}")

        # Diseminación
        if self.population > 1000:
            self.diseminar()

    def diseminar(self):
        x, y = self.pos
        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.model.width and 0 <= ny < self.model.height:
                    cell_clones = [a for a in self.model.grid[nx][ny] if isinstance(a, KPNClone)]
                    total_pop = sum(c.population for c in cell_clones)
                    if total_pop < self.population * 0.5:
                        candidates.append((nx, ny, total_pop))

        if candidates:
            candidates.sort(key=lambda c: c[2])
            nx, ny, _ = candidates[0]
            trans_amount = int(self.population * 0.1)
            cell_clones = [a for a in self.model.grid[nx][ny] if isinstance(a, KPNClone)]
            if not cell_clones:
                new_clone = KPNClone(unique_id=f"{self.unique_id}_migrate", model=self.model)
                new_clone.state = self.state
                new_clone.population = trans_amount
                new_clone.pos = (nx, ny)
                self.model.agents.append(new_clone)
                self.model.grid[nx][ny].append(new_clone)
                print(f"[DISEMINATION] Nuevo clon creado: {new_clone.unique_id} en {new_clone.pos} con pobl {new_clone.population}")
            else:
                chosen_clone = random.choice(cell_clones)
                chosen_clone.population += trans_amount
                print(f"[DISEMINATION] KPNClone {self.unique_id} transfiere {trans_amount} población a {chosen_clone.unique_id}")

            self.population -= trans_amount


class HealthcareWorker:
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
        print(f"[INIT] HealthcareWorker {self.unique_id} creado en pos {self.pos}, inm {self.nivel_inmunidad}")

    def step(self):
        print(f"[STEP] HealthcareWorker {self.unique_id} en {self.pos}, portador={self.portador_kpn}")
        if self.portador_kpn:
            self.move_to_less_crowded()
        else:
            if random.random() < 0.3:
                self.move_towards_contaminated_equipment()
            else:
                self.random_move()

        if self.portador_kpn:
            self.recovery_steps_remaining -= 1
            if self.recovery_steps_remaining <= 0:
                print(f"[RECOVERY] HealthcareWorker {self.unique_id} se recupera.")
                self.portador_kpn = False
                self.reinfection_resistance += 0.05

        cellmates = self.model.get_cell_list_contents(self.pos)
        base_transmission_prob = 0.15 if random.random() < self.prob_bioseguridad else 0.35
        if self.extra_barrier:
            base_transmission_prob *= 0.5

        for obj in cellmates:
            if obj is not self:
                if hasattr(obj, "portador_kpn") or hasattr(obj, "contaminado"):
                    # Check contagio bidireccional
                    if self.portador_kpn and isinstance(obj, Patient) and random.random() < base_transmission_prob:
                        print(f"[TRANSMISSION] HealthcareWorker {self.unique_id} infecta a Patient {obj.unique_id}")
                        obj.set_infection()
                    elif self.portador_kpn and isinstance(obj, ReusableEquipment) and random.random() < base_transmission_prob:
                        print(f"[TRANSMISSION] HealthcareWorker {self.unique_id} contamina equipo {obj.unique_id}")
                        obj.contaminado = True

                    if getattr(obj, "portador_kpn", False) or getattr(obj, "contaminado", False):
                        # Riesgo de que el worker se infecte
                        if random.random() < (base_transmission_prob * (1 - self.reinfection_resistance)):
                            print(f"[INFECTION] HealthcareWorker {self.unique_id} se infecta a partir de {obj.unique_id}")
                            self.set_infection()

    def set_infection(self):
        self.portador_kpn = True
        self.recovery_steps_remaining = random.randint(5, 15)

    def move_towards_contaminated_equipment(self):
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
            print(f"[MOVE] HealthcareWorker {self.unique_id} va hacia equipo contaminado en {best_cell}")
            self.move_to(best_cell)
        else:
            self.random_move()

    def move_to_less_crowded(self):
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
            print(f"[MOVE] HealthcareWorker {self.unique_id} se mueve a celda menos poblada {best_cell}")
            self.move_to(best_cell)
        else:
            self.random_move()

    def move_to(self, new_pos):
        x,y = self.pos
        self.model.grid[x][y].remove(self)
        self.model.grid[new_pos[0]][new_pos[1]].append(self)
        self.pos = new_pos
        print(f"[MOVE] HealthcareWorker {self.unique_id} movido a {self.pos}")

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
            print(f"[MOVE] HealthcareWorker {self.unique_id} se mueve aleatoriamente a {new_pos}")
            self.move_to(new_pos)


class Patient:
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
        print(f"[INIT] Patient {self.unique_id} creado con inm {self.nivel_inmunidad}, portador={self.portador_kpn}")

        if self.portador_kpn:
            self.recovery_steps_remaining = 10

    def step(self):
        print(f"[STEP] Patient {self.unique_id} en {self.pos}, portador={self.portador_kpn}")
        if self.portador_kpn:
            if random.random() < 0.3:
                self.move_towards_hitech_equipment()
            else:
                self.random_move()
        else:
            if random.random() < 0.3:
                self.move_away_from_contamination()
            else:
                self.random_move()

        if self.portador_kpn:
            self.recovery_steps_remaining -= 1
            if self.recovery_steps_remaining <= 0:
                print(f"[RECOVERY] Patient {self.unique_id} se recupera.")
                self.portador_kpn = False

        cellmates = self.model.get_cell_list_contents(self.pos)
        base_trans_prob = 0.25
        if self.en_aislamiento:
            base_trans_prob *= 0.5

        for obj in cellmates:
            if obj is not self:
                if isinstance(obj, HealthcareWorker) or isinstance(obj, Patient):
                    if self.portador_kpn and random.random() < base_trans_prob:
                        print(f"[TRANSMISSION] Patient {self.unique_id} infecta a {obj.unique_id}")
                        obj.set_infection()
                    elif getattr(obj, "portador_kpn", False) and random.random() < base_trans_prob:
                        print(f"[INFECTION] Patient {self.unique_id} se infecta a partir de {obj.unique_id}")
                        self.set_infection()

                if isinstance(obj, ReusableEquipment):
                    if self.portador_kpn and random.random() < base_trans_prob:
                        print(f"[CONTAMINATE] Patient {self.unique_id} contamina equipo {obj.unique_id}")
                        obj.contaminado = True
                    elif obj.contaminado and random.random() < base_trans_prob:
                        print(f"[INFECTION] Patient {self.unique_id} se infecta por equipo {obj.unique_id}")
                        self.set_infection()

        if self.portador_kpn:
            is_resistant = (random.random() < 0.5)
            if is_resistant:
                if self.recibe_antibiotico:
                    self.terapia_combinada = True
                    if random.random() < 0.1:
                        print(f"[ANTIBIOTICS] Patient {self.unique_id} se cura con terapia combinada")
                        self.portador_kpn = False
                        self.recovery_steps_remaining = 0
            else:
                if self.recibe_antibiotico:
                    if random.random() < 0.3:
                        print(f"[ANTIBIOTICS] Patient {self.unique_id} se cura con antibiótico normal")
                        self.portador_kpn = False
                        self.recovery_steps_remaining = 0

    def set_infection(self):
        self.portador_kpn = True
        self.recovery_steps_remaining = 10

    def move_away_from_contamination(self):
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
            print(f"[MOVE] Patient {self.unique_id} alejarse de contaminación hacia {best_cell}")
            self.move_to(best_cell)
        else:
            self.random_move()

    def move_towards_hitech_equipment(self):
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
            print(f"[MOVE] Patient {self.unique_id} hacia equipo alta-tec en {new_pos}")
            self.move_to(new_pos)
        else:
            self.random_move()

    def move_to(self, new_pos):
        x,y = self.pos
        self.model.grid[x][y].remove(self)
        self.model.grid[new_pos[0]][new_pos[1]].append(self)
        self.pos = new_pos
        print(f"[MOVE] Patient {self.unique_id} movido a {self.pos}")

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
            print(f"[MOVE] Patient {self.unique_id} se mueve aleatoriamente a {new_pos}")
            self.move_to(new_pos)


class ReusableEquipment:
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
        print(f"[INIT] ReusableEquipment {self.unique_id}, alta_tecnologia={self.alta_tecnologia}, contaminado={self.contaminado}")

    def step(self):
        cellmates = self.model.get_cell_list_contents(self.pos)
        if any(isinstance(a, HealthcareWorker) for a in cellmates):
            effective_cleanup = self.cleanup_prob + 0.05
        else:
            effective_cleanup = self.cleanup_prob

        if self.contaminado and random.random() < effective_cleanup:
            print(f"[CLEANUP] ReusableEquipment {self.unique_id} se limpia.")
            self.contaminado = False
            self.tiempo_contaminado = 0

        if self.contaminado:
            self.tiempo_contaminado += 1
            self.biofilm_rate = min(self.biofilm_rate + 0.001, 0.3)
            print(f"[BIOFILM] ReusableEquipment {self.unique_id} contaminado por {self.tiempo_contaminado} pasos, biofilm_rate={self.biofilm_rate}")
