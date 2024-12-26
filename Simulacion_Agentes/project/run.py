import matplotlib.pyplot as plt
import numpy as np
from model import ICUSimulation

def single_run_simulation(params):
    print("[RUN] Iniciando simulación única.")
    model = ICUSimulation(**params)
    while model.running:
        model.step()
    return model.datacollector, model.current_step, model

def multi_run_simulation(n_runs=10):
    print("[RUN] Ejecutando múltiples réplicas.")
    params = {
        'width': 200,
        'height': 200,
        'initial_kpn_population': 5000,
        'n_patients': 300,
        'n_workers': 30,
        'n_equipment': 10,
        'n_clones': 10,
        'max_steps': 700
    }

    runs_data = []
    max_steps_encountered = 0

    for run_idx in range(n_runs):
        print(f"[RUN] Réplica {run_idx+1}/{n_runs}")
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

    print("[RUN] Graficando resultados agregados.")
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
    multi_run_simulation(n_runs=5)

if __name__ == "__main__":
    run_simulation_multiple_replicas()
