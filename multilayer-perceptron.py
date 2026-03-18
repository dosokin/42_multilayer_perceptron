#%%
from platform import architecture
!pip install pandas
!pip install matplotlib
!pip install seaborn
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
try:
    df = pd.read_csv("./data.csv")
except Exception as e:
    print("Error reading the data source file: e")
    quit()
#%%
df.columns = [
    "id",
    "diagnostic",

    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_largest",
    "texture_largest",
    "perimeter_largest",
    "area_largest",
    "smoothness_largest",
    "compactness_largest",
    "concavity_largest",
    "concave_points_largest",
    "symmetry_largest",
    "fractal_dimension_largest",
]

df.head(3)
#%%
df = df.drop(['id'], axis=1)
df.head(3)
#%%
df['diagnostic'] = df['diagnostic'].str.replace("B", "0")
df['diagnostic'] = df['diagnostic'].str.replace("M", "1")
df['diagnostic'] = pd.to_numeric(df['diagnostic'])
df.head(3)
#%%
diagnostic_corr = df.corr()['diagnostic'].sort_values()
diagnostic_corr
#%%
diagnostic_corr.sort_values(ascending=False).head(10)
#%%
diagnostic_corr.sort_values().head(5)
#%%
sns.pairplot(
    df,
    x_vars=['diagnostic'],
    y_vars=['concave_points_largest', 'perimeter_largest', 'concave_points_mean'],
    kind='hist'
)
#%%
sns.pairplot(
    df,
    x_vars=['diagnostic'],
    y_vars=['fractal_dimension_se'],
    kind='hist'
)
#%%
filtered_df = df[['diagnostic',
        'concave_points_largest',
        'perimeter_largest',
        'concave_points_mean',
        'radius_largest',
        'perimeter_mean',
        'area_largest',
        'radius_mean',
        'area_mean',
        'concavity_mean']]

df = df[[
    "diagnostic",
    "concavity_se",
    "compactness_se",
    "fractal_dimension_largest",
    "symmetry_mean",
    "smoothness_mean",
    "concave_points_se",
    "symmetry_largest",
    "smoothness_largest",
    "texture_mean",
    "texture_largest",
    "area_se",
    "perimeter_se",
    "radius_se",
    "compactness_largest",
    "compactness_mean",
    "concavity_largest",
    "concavity_mean",
    "area_mean",
    "radius_mean",
    "area_largest",
    "perimeter_mean",
    "radius_largest",
    "concave_points_mean",
    "perimeter_largest",
    "concave_points_largest"
]]

df.head(3)

#%%
for col in df.columns:
  df[col] = pd.to_numeric(df[col])
  df[col] = df[col] / df[col].max()
df.head()
#%%
for col in filtered_df.columns:
  filtered_df[col] = pd.to_numeric(df[col])
  filtered_df[col] = filtered_df[col] / filtered_df[col].max()
#%%

np.random.seed(0)

mask = np.random.rand(len(df)) < 0.8

train = df[mask].to_numpy().astype(np.float64)
test = df[~mask].to_numpy().astype(np.float64)

print(len(test))
print(len(train))
#%%
filtered_train = filtered_df[mask].to_numpy().astype(np.float64)
filtered_test = filtered_df[~mask].to_numpy().astype(np.float64)
#%%
def softmax(o):
  def inner_softmax(x):
    return x / o.sum()
  softed_o = inner_softmax(o)
  return softed_o
#%%
def get_loss_cee(predictions, expected):
  def inner_cee(p, y):
    return y * np.log(p) + (1 - y) * np.log(1 - p)
  loss = inner_cee(predictions, expected)
  loss = loss.sum() * -1 / len(predictions)

  return loss
#%%
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def derivated_sigmoid(x):
  return sigmoid(x) * (1 - sigmoid(x))

#%%
class Layer:

  def __init__(self, neurons_count, input_size, previous_layer=None, parameters=None):
    self.neurons_count = neurons_count

    if parameters:
        self.W = np.array(parameters["weights"])
        self.b = np.array(parameters["biases"])
    else:
        self.W = np.random.rand(neurons_count, input_size)
        self.b = np.zeros(neurons_count).reshape(neurons_count, 1)

    self.z = None
    self.A = None
    self.C = None

    self.previous_layer = previous_layer
    if self.previous_layer:
      self.previous_layer.next_layer = self

    self.next_layer = None
    self.incoming_A = None
    self.delta = None

  def forward_pass(self, a, result=False):
    self.incoming_A = a

    self.z = np.matmul(self.W, self.incoming_A) + self.b

    self.A = sigmoid(self.z)
    if self.next_layer:
      return self.next_layer.forward_pass(self.A, result=result)

    return softmax(self.A)

  def set_delta(self, expected_y=None):

    if self.next_layer is None:
      dL = 2 * (self.A - expected_y)
      self.delta = dL * derivated_sigmoid(self.z)

    else:
      self.delta = np.transpose(self.next_layer.W) @ self.next_layer.delta * derivated_sigmoid(self.z)

    if self.previous_layer:
      self.previous_layer.set_delta()

  def set_c(self):
    self.C = self.delta @ np.transpose(self.incoming_A)
    if self.previous_layer:
      self.previous_layer.set_c()


  def adjust_weights(self):
    self.W = self.W - (LEARNING_RATE * self.C)
    if self.previous_layer:
      self.previous_layer.adjust_weights()

  def adjust_biases(self):
      self.b = self.b - (LEARNING_RATE * self.delta)
      if self.previous_layer:
          self.previous_layer.adjust_biases()

  def get_parameters(self):

        neurons = []

        for (w, b) in zip(self.W, self.b):
            neurons.append({
                "weights": w.tolist(),
                "biases": b.tolist()
            })

        return {
            "neurons_count": self.neurons_count,
            "neurons": neurons
        }


class NN:

    def load_model(self, model):

        self.architecture = model['architecture']['layers_architecture']
        self.features_count = model['architecture']['features_count']

        self.layers = []
        current_layer = None
        for l in model["layers"]:
            current_layer = Layer(neurons_count=l["neurons_count"],
                                  input_size=len(l["neurons"][0]),
                                  previous_layer=current_layer,
                                  parameters=l["neurons"])
            self.layers.append(current_layer)

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]


    def __init__(self, *args, model_to_load=None):
        if model_to_load:
            self.load_model(model_to_load)
            return

        if len(args) <= 1:
            raise Exception("The neural network should have at least one layer")

        self.features_count = int(args[0])
        self.architecture = [int(x) for x in args[1:]]

        self.layers = []

        current_layer = None
        for i, neurons_count in enumerate(self.architecture):
            current_layer = Layer (
                neurons_count=neurons_count,
                input_size=args[i],
                previous_layer=current_layer,
            )
            self.layers.append(current_layer)

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]


    def init_forward_pass(self, x):
        return self.input_layer.forward_pass(x)


    def init_back_propagation(self, expected_y):
        self.output_layer.set_delta(expected_y=expected_y)
        self.output_layer.set_c()
        self.output_layer.adjust_weights()
        self.output_layer.adjust_biases()


    def get_model(self):
        model = {
            "architecture": {
                "features_count": self.features_count,
                "output_count": self.architecture[-1],
                "layers_architecture": self.architecture
            },
            "layers_count": len(self.layers),
            "layers": []
        }

        for l in self.layers:
            model["layers"].append(l.get_parameters())

        return model
#%%
MAX_EPOCH = 10000
PATIENCE = 50
WARMING_EPOCH = 100
LEARNING_RATE = 0.01
#%%
FEATURES_COUNT = len(train[0]) - 1
EPOCH_COUNT = 0

try:
    nn = NN(FEATURES_COUNT,16,8,2)
except Exception as e:
    print(e)
    quit()

best_model = {
    "loss": None,
    "model": None,
}

len_train = len(train)
len_test = len(test)

evolution_accuracy = []
evolution_train_loss = []
evolution_test_loss = []

p = 0

for x in range(MAX_EPOCH):
    cumulate_train_loss = 0
    for i, v in enumerate(train):
        X = np.reshape(v[1:], (FEATURES_COUNT, 1))
        r = nn.init_forward_pass(X)
        nn.init_back_propagation(np.array([[v[0]], [1 - v[0]]]))
        cumulate_train_loss += get_loss_cee(r, np.array([[v[0]], [1 - v[0]]]))
    cumulate_test_loss = 0
    epoch_accuracy = 0
    for i, v in enumerate(test):
        X = np.reshape(v[1:], (FEATURES_COUNT, 1))
        r = nn.init_forward_pass(X)
        cumulate_test_loss += get_loss_cee(r, np.array([[v[0]], [1 - v[0]]]))
        if bool(r[0] < r[1]) ^ bool(v[0]):
            epoch_accuracy += 1

    epoch_accuracy = epoch_accuracy / len_test
    train_loss = cumulate_train_loss / len_train
    test_loss = cumulate_test_loss / len_test

    evolution_accuracy.append(epoch_accuracy)
    evolution_train_loss.append(train_loss)
    evolution_test_loss.append(test_loss)

    EPOCH_COUNT += 1

    if best_model["loss"] is None or train_loss < best_model["loss"]:
        best_model["loss"] = train_loss
        best_model["model"] = nn.get_model()
        p = 0

    elif x > WARMING_EPOCH and train_loss > best_model["loss"] + 0.01:
        p += 1

    if p > PATIENCE:
        break

    if x % 100 == 0:
        print(f"train_L = {train_loss:.3f}  val_L = {test_loss:.3f} acc = {epoch_accuracy}")

#%%
# FEATURES_COUNT = len(filtered_train[0]) - 1
#
# firstNN = NN(
#     FEATURES_COUNT,
#     16,
#     8,
#     2
# )
#
# len_train = len(filtered_train)
# len_test = len(filtered_test)
#
# filtered_evolution_accuracy = []
# filtered_evolution_train_loss = []
# filtered_evolution_test_loss = []
#
# for x in range(MAX_EPOCH):
#     train_loss = 0
#     for i, v in enumerate(filtered_train):
#         X = np.reshape(v[1:], (FEATURES_COUNT, 1))
#         r = firstNN.init_forward_pass(X)
#         firstNN.init_back_propagation(np.array([[v[0]], [1 - v[0]]]))
#         train_loss += get_loss_cee(r, np.array([[v[0]], [1 - v[0]]]))
#     test_loss = 0
#     epoch_accuracy = 0
#     for i, v in enumerate(filtered_test):
#         X = np.reshape(v[1:], (FEATURES_COUNT, 1))
#         r = firstNN.init_forward_pass(X)
#         test_loss += get_loss_cee(r, np.array([[v[0]], [1 - v[0]]]))
#         if bool(r[0] < r[1]) ^ bool(v[0]):
#             epoch_accuracy += 1
#
#     filtered_evolution_accuracy.append(epoch_accuracy)
#     filtered_evolution_train_loss.append(train_loss / len_train)
#     filtered_evolution_test_loss.append(test_loss / len_test)
#
#     # if x % 100 == 0:
#     #     print(f"train_L = {train_loss / len_train:.3f}  val_L = {test_loss / len_test:.3f} acc = {epoch_accuracy} /  {len_test}")

#%%
plt.plot(np.arange(1, EPOCH_COUNT + 1, 1), np.array(evolution_accuracy), color='r')
# plt.plot(np.arange(1, MAX_EPOCH + 1, 1), np.array(filtered_evolution_accuracy), color='b')
#%%
plt.plot(np.arange(1, EPOCH_COUNT + 1, 1), np.array(evolution_train_loss), 'r')
plt.plot(np.arange(1, EPOCH_COUNT + 1, 1), np.array(evolution_test_loss), 'r--')
# plt.plot(np.arange(1, MAX_EPOCH + 1, 1), np.array(filtered_evolution_train_loss), color='yellow')
# plt.plot(np.arange(1, MAX_EPOCH + 1, 1), np.array(filtered_evolution_test_loss), color='orange')
plt.show()
#%%
from pathlib import Path
import json

def save_model(filename, model=None, nn=None):
    if not model:
        if nn:
            model = nn.get_model()
        else:
            raise Exception("Defined model or NN object required to save the model")

    output_file = Path(f"./models/{filename}.txt")

    if not output_file.exists():
        output_file.parent.mkdir(exist_ok=True, parents=True)

    print(f"SAVING MODEL INTO {output_file.absolute()}")

    try:
        with output_file.open('w') as f:
            json.dump(model, f)
    except Exception as e:
        raise Exception(f"Error writing into {output_file.absolute()}: {e}")
#%%
print(best_model)
#%%
save_model("test", best_model)
#%%
