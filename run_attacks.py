import subprocess


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # print the output
    for line in iter(process.stdout.readline, b''):
        print(line.decode(), end='')

    # wait for the process ending
    process.stdout.close()
    return_code = process.wait()
    if return_code == 0:
        print("Command ran successfully")
    else:
        print(f"Command failed with return code {return_code}")


for i in range(1):
    clean_shadow_model = [
        "rm", "./shadow_1_c10/checkpoint.pth"
    ]

    run_command(clean_shadow_model)

    train_new_shadow_model = [
        "python", "./training/train_shadow_model.py",
        "--dump_path", "./shadow_1_c10/",
        "--mask_path", "./data/",
        "--dataset", "cifar10",
        "--shadow_model", "0",
        "--optimizer", "sgd,lr=0.01,momentum=0.9",
        "--epochs", "20",
        "--architecture", "vgg16",
        "--batch_size", "64"
    ]

    run_command(train_new_shadow_model)

    train_new_shadow_model_1 = [
        "python", "./training/train_shadow_model.py",
        "--dump_path", "./shadow_1_c10/",
        "--mask_path", "./data/",
        "--dataset", "cifar10",
        "--shadow_model", "1",    # if shadow_model == 0, retrain the shadow model
        "--optimizer", "sgd,lr=0.004,momentum=0.9",
        "--epochs", "25",
        "--architecture", "vgg16",
        "--batch_size", "64"
    ]

    run_command(train_new_shadow_model_1)

    clean_reference_model = [
        "rm", "./attack_model_c10/checkpoint.pth"
    ]

    run_command(clean_reference_model)

    multi_flag = "1"
    if i == 0:
        multi_flag = "0"

    train_reference_model = [
        "python", "MIA_classifier.py",
        "--dump_path", "./shadow_1_c10/",
        "--mask_path", "./data/",
        "--dataset", "cifar10",
        "--multi", multi_flag,   # if multi == 1, only retrain reference model, if multi == 0, also train classifier
        "--optimizer", "sgd,lr=0.01,momentum=0.9",
        "--attack_architecture", "vgg16",
        "--private_architecture", "vgg16",
        "--cos_threshold", "0",
        "--classifier_epochs", "35",
        "--batch_size", "64",
        "--classifier_num_dimensions", "12",
        "--dump_path", "attack_model_c10",
        "--shadow_1_path", "shadow_1_c10",
        "--num_classes", "10",
        "--mia_model_path", "mia_model_c10",
        "--epochs", "30"
    ]

    run_command(train_reference_model)

