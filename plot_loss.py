import torch
import matplotlib.pyplot as plt

data = torch.load("models/loss_log.pt")

plt.plot(data["iter"], data["loss"])
plt.xlabel("iteration_number")
plt.ylabel("training_loss")
plt.title("Aimonk_multilabel_problem")
plt.savefig("models/loss_curve.png", dpi=300)
plt.show()