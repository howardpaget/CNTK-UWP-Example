# Adapted from this example: https://cntk.ai/pythondocs/CNTK_101_LogisticRegression.html

import cntk
import numpy as np

input_dim = 2
num_output_classes = 1

np.random.seed(0)

# Create features and labels that are dependent on them
features = np.asarray(np.random.random_sample((10000, 2)), dtype=np.float32)
labels = np.asarray([np.asarray([1 if 1 / (1 + np.exp(-(.2 * x[0] + .3 * x[1] - .5))) >= 0.5 else 0], np.float32) for x in features], dtype=np.float32)
#labels = np.asarray([np.asarray([1 / (1 + np.exp(-(.2 * x[0] + .3 * x[1] - .5)))], np.float32) for x in features], dtype=np.float32)

# Create the model, label = sigmoid(feature * W + b)
def create_model(input_var, output_dim):
    weight = cntk.parameter(shape=(input_var.shape[0], output_dim), name='W')
    bias = cntk.parameter(shape=(output_dim), name='b')
    
    return cntk.sigmoid(cntk.times(input_var, weight) + bias, name='o')

feature = cntk.input_variable(input_dim, np.float32)
model = create_model(feature, num_output_classes)

# Set up inputs and functions used by the trainer
label = cntk.input_variable(num_output_classes, np.float32)
loss = cntk.squared_error(model, label)
eval_error = cntk.squared_error(model, label)

# Create the trianer using a stochastic gradient descent (sgd) learner
learning_rate = 0.5
lr_schedule = cntk.learning_parameter_schedule(learning_rate)
learner = cntk.sgd(model.parameters, lr_schedule)
trainer = cntk.Trainer(model, (loss, eval_error), [learner])

# Fit the model
for i in range(1000):
    trainer.train_minibatch({feature: features, label: labels})

    if i % 100 == 0:
        print ('Batch: {0}, Loss: {1:.4f}, Error: {2:.2f}'.format(i, trainer.previous_minibatch_loss_average, trainer.previous_minibatch_evaluation_average))

# Save the model for later import into UWP app
model.save('../CNTKUWPApp/model.model')

# Evaluate the model on the training data
result = model.eval({feature : features})
predicted = [np.asarray([1], np.float32) if r >= 0.5 else np.asarray([0], np.float32) for r in result]
#predicted = [np.asarray([r], np.float32) for r in result]

comparison = np.abs(labels - predicted)

print('% Correct: {0:.4f}'.format(100 * (1 - np.sum(comparison) / comparison.shape[0])))

print('W: ', model.parameters[0].value)
print('b: ', model.parameters[1].value)
