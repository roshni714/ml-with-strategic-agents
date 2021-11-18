import csv
import os


def report_results(results, thetas, losses, save):
    results_file = "results/" + save + ".csv"
    write_result(results_file, results)

    train_file = "results/" + save + "_train.csv"

    for i in range(len(thetas)):
        iteration_dic = {}
        iteration_dic["iteration"] = i
        iteration_dic["loss"] = losses[i]
        iteration_dic["theta"] = thetas[i]
        write_result(train_file, iteration_dic)


def write_result(results_file, result):
    """Writes results to a csv file."""
    with open(results_file, "a+", newline="") as csvfile:
        field_names = result.keys()
        dict_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if os.stat(results_file).st_size == 0:
            dict_writer.writeheader()
        dict_writer.writerow(result)
