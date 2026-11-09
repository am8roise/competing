import numpy as np

list = np.round(np.linspace(0, np.pi/2, 51), 15)

for element in list:
        with open("./inputs.txt", 'a') as file:
                file.write(f"{element}\n")