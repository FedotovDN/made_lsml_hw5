import breeze.linalg._

print("Reading data...")
val bufferedSource = io.Source.fromFile("C:\\DB\\diabet.csv")
var i = 0
//We have 442 rows in our data. Let's put 400 rows in train, and other - in validation
val X_train = DenseMatrix.zeros[Double](400, 10)
val y_train = DenseVector.zeros[Double](400)
val X_test = DenseMatrix.zeros[Double](42, 10)
val y_test = DenseVector.zeros[Double](42)
for (line <- bufferedSource.getLines) {
  val cols = line.split(",").map(_.trim)
  //splitting on train/test dataset
  if (i < 400) {
    for (j <- 0 to 9) {
      X_train(i, j) = cols(j).toDouble: Double
    }
    y_train(i) = cols(10).toDouble: Double
  } else {
    for (j <- 0 to 9) {
      X_test(i - 400, j) = cols(j).toDouble: Double
    }
    y_test(i - 400) = cols(10).toDouble: Double
  }
  i = i + 1
}
bufferedSource.close

//Counting coefficients
val coef = pinv(X_train.t * X_train) * X_train.t * y_train
println("Coefficients:")
println(coef)

//And now check our coefficients on train-dataset
var s = 0.0
var err_value = 0.0
var err_sum = 0.0
//введем значение intercept для линейной регрессии (оценочно)
var intercept = 150.0
for (i <- 0 to 399) {
  s = 0.0
  for (j <- 0 to 9) {
    s = s + coef(j) * X_train(i, j)
  }
  s = s + intercept
  err_value = s - y_train(i)
  err_sum = err_sum + err_value.abs
}
println("MAE on TRAIN-dataset:")
println(err_sum / 400)

//And now check our coefficients on test-dataset
err_sum = 0.0
for (i <- 0 to 41) {
  s = 0.0
  for (j <- 0 to 9) {
    s = s + coef(j) * X_test(i, j)
  }
  s = s + intercept
  err_value = s - y_test(i)
  err_sum = err_sum + err_value.abs
}
println("MAE on TEST-dataset:")
println(err_sum / 42)