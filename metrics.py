import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

models = os.listdir("../checkpoints/try3")
models.remove("output.log")

models = [i.split("_") for i in models]
df = pd.DataFrame(models, columns=["epoch", "loss",  "acc"])
df["epoch"] = df["epoch"].apply(lambda i: i[5:]).astype(int)
df["loss"] = df["loss"].apply(lambda i: i[4:]).astype(float)
df["acc"] = df["acc"].apply(lambda i: i[3:-4]).astype(float)
df = df.sort_values(by=["epoch"], ignore_index=True)

plt.figure(1)
plt.plot(df["epoch"][20:], df["loss"][20:], label="loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("Loss.png")

plt.figure(2)
plt.plot(df["epoch"][20:], df["acc"][20:])
plt.title("Validation Accuraccy (%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("Acc.png")
