import tensorflow as tf
df_train = "/home/mdabis/szakdolgozat/models/99_regression/boston_train.csv"
df_eval = "/home/mdabis/szakdolgozat/models/99_regression/boston_test.csv"
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
                "dis", "tax", "ptratio", "medv"]
RECORDS_ALL = [[0.0], [0.0], [0.0], [0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]

def input_fn(data_file, batch_size, num_epoch = None):
       # Step 1
      def parse_csv(value):
          columns = tf.decode_csv(value, record_defaults= RECORDS_ALL)
          features = dict(zip(COLUMNS, columns))
          #labels = features.pop('median_house_value')
          labels =  features.pop('medv')
          return features, labels

      # Extract lines from input files using the Dataset API.
      dataset = (tf.data.TextLineDataset(data_file) # Read text file
      .skip(1) # Skip header row
      .map(parse_csv))

      dataset = dataset.repeat(num_epoch)
      dataset = dataset.batch(batch_size)
      # Step 3
      iterator = dataset.make_one_shot_iterator()
      features, labels = iterator.get_next()
      return features, labels

def test_input_fn():
    dataset = tf.data.Dataset.from_tensors(prediction_input)
    return dataset

next_batch = input_fn(df_train, batch_size = 1, num_epoch = None)
with tf.Session() as sess:
     first_batch  = sess.run(next_batch)
     print(first_batch)

X1= tf.feature_column.numeric_column('crim')
X2= tf.feature_column.numeric_column('zn')
X3= tf.feature_column.numeric_column('indus')
X4= tf.feature_column.numeric_column('nox')
X5= tf.feature_column.numeric_column('rm')
X6= tf.feature_column.numeric_column('age')
X7= tf.feature_column.numeric_column('dis')
X8= tf.feature_column.numeric_column('tax')
X9= tf.feature_column.numeric_column('ptratio')

base_columns = [X1, X2, X3,X4, X5, X6,X7, X8, X9]

model = tf.estimator.LinearRegressor(feature_columns=base_columns, model_dir='train3')

model.train(steps =1000, input_fn= lambda : input_fn(df_train,batch_size=128, num_epoch = None))

model.save('my_model')

results = model.evaluate(steps =None,input_fn=lambda: input_fn(df_eval, batch_size =128, num_epoch = 1))
for key in results:
    print("   {}, was: {}".format(key, results[key]))


prediction_input = {
          'crim': [0.03359,5.09017,0.12650,0.05515,8.15174,0.24522],
          'zn': [75.0,0.0,25.0,33.0,0.0,0.0],
          'indus': [2.95,18.10,5.13,2.18,18.10,9.90],
          'nox': [0.428,0.713,0.453,0.472,0.700,0.544],
          'rm': [7.024,6.297,6.762,7.236,5.390,5.782],
          'age': [15.8,91.8,43.4,41.1,98.9,71.7],
          'dis': [5.4011,2.3682,7.9809,4.0220,1.7281,4.0317],
          'tax': [252,666,284,222,666,304],
          'ptratio': [18.3,20.2,19.7,18.4,20.2,18.4]
     }


# Predict all our prediction input
pred_results = model.predict(input_fn=test_input_fn)

for pred in enumerate(pred_results):
    print(pred)
