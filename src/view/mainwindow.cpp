#include "mainwindow.h"

#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      ui_(new Ui::MainWindow),
      controller_(new s21::Controller) {
  ui_->setupUi(this);
  ui_->WidgetForPainting->SetWindow(this);
  graph_ = new Graph(ui_->graph);

  ConnectSignals();
}

MainWindow::~MainWindow() {
  delete label_;
  delete ui_;
  delete graph_;
}

void MainWindow::ConnectSignals() {
  connect(this, &MainWindow::signalMetrics, this, &MainWindow::ExperimentOver);
  connect(this, &MainWindow::signalMetrics, graph_, &Graph::Draw);
  connect(this, &MainWindow::signalTestProgress, this,
          &MainWindow::UpdateTestBar);
  connect(this, &MainWindow::signalTrainProgress, this,
          &MainWindow::UpdateTrainBar);
  connect(this, &MainWindow::signalFullTrainProgress, this,
          &MainWindow::UpdateFullTrainBar);


  connect(ui_->LoadWeights, SIGNAL(clicked()), this, SLOT(LoadWeightsClicked()));
  connect(ui_->LoadPicture, SIGNAL(clicked()), this, SLOT(LoadPictureClicked()));
  connect(ui_->SaveWeights, SIGNAL(clicked()), this, SLOT(SaveWeightsClicked()));
  connect(ui_->LoadDataTraining, SIGNAL(clicked()), this, SLOT(LoadDataTrainingClicked()));
  connect(ui_->LoadDataExperiment, SIGNAL(clicked()), this, SLOT(LoadDataExperimentClicked()));
  connect(ui_->PerceptronTypeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(PerceptronTypeChanged(int)));
  connect(ui_->HiddenLayers, SIGNAL(currentTextChanged(QString)), this, SLOT(HiddenLayersChanged(QString)));
  connect(ui_->RunTesting, SIGNAL(clicked()), this, SLOT(RunTestingClicked()));
  connect(ui_->RunExperiment, SIGNAL(clicked()), this, SLOT(RunExperimentClicked()));
  connect(ui_->Result, SIGNAL(clicked()), this, SLOT(PredictClicked()));
}

void MainWindow::LoadPictureClicked() {
  QString path = QFileDialog::getOpenFileName(this, "Load from file..",
                                              QDir::homePath(), "(*.bmp)");
  try {
    if (!path.isEmpty()) {
      if (label_) label_->clear();
      label_ = new QLabel("");
      label_->resize(ui_->WidgetForPainting->width(),
                     ui_->WidgetForPainting->height());
      label_->setPixmap(QPixmap(path).scaled(ui_->WidgetForPainting->width(),
                                             ui_->WidgetForPainting->height()));

      ui_->WidgetForPainting->setLayout(new QVBoxLayout);
      ui_->WidgetForPainting->layout()->addWidget(label_);
      flag_load_picture_ = 1;
      NormalizePicture(QImage(path));
    }
  } catch (std::runtime_error &exept) {
    ShowExeption(exept.what());
  }
}

void MainWindow::NormalizePicture(QImage picture) {
  vector_with_pix_.clear();
  QImage scaledPicture = picture.scaled(28, 28, Qt::KeepAspectRatio);
  for (auto i = 0; i < 28; ++i) {
    for (auto j = 0; j < 28; ++j) {
      vector_with_pix_.push_back(scaledPicture.pixelColor(i, j).blackF());
    }
  }
}

void MainWindow::PredictClicked() {
  try {
    if (!vector_with_pix_.empty()) {
      QString str = "";
      str += controller_->GetPredict(vector_with_pix_);
      ui_->ResultingLetter->setText(str);
    }
  } catch (std::runtime_error &exept) {
    ShowExeption(exept.what());
    ClearResultingLetter();
  }
}

void MainWindow::ClearResultingLetter() { ui_->ResultingLetter->clear(); }

void MainWindow::PerceptronTypeChanged(int index) {
  controller_->SetType(index);
}

void MainWindow::LoadWeightsClicked() {
  QString path = QFileDialog::getOpenFileName(this, "Load from file..",
                                              QDir::homePath(), "(*.bin)");
  try {
    if (!path.isEmpty()) {
      controller_->LoadWeights(path.toStdString());
      ui_->NumOfHiddenLayers->setText(
          QString::number(controller_->GetHiddenLayersNum()));
    }
  } catch (std::runtime_error &exept) {
    ShowExeption(exept.what());
  }
}

void MainWindow::SaveWeightsClicked() {
  QString path = QFileDialog::getSaveFileName(this, "Save to..",
                                              QDir::homePath(), "(*.bin)");
  try {
    if (!path.isEmpty()) {
      controller_->SaveWeights(path.toStdString());
    }
  } catch (std::runtime_error &exept) {
    ShowExeption(exept.what());
  }
}

void MainWindow::LoadDataTrainingClicked() { LoadFile(true); }

void MainWindow::LoadDataExperimentClicked() { LoadFile(false); }

void MainWindow::LoadFile(bool is_train) {
  QString path = QFileDialog::getOpenFileName(this, "Load from file..",
                                              QDir::homePath(), "(*.csv)");
  try {
    if (!path.isEmpty()) {
      if (is_train)
        controller_->SetTrainDataset(path.toStdString());
      else
        controller_->SetExperimentDataset(path.toStdString());

      QFileInfo name(path);
      if (is_train) {
        ui_->FilenameDataTraining->setText(name.fileName());
        ui_->SizeDataTraining->setText(
            QString::number(controller_->GetSizeOfTrainDataset()));
        file_train_ = 1;
      } else {
        ui_->FilenameDataExperiment->setText(name.fileName());
        ui_->SizeDataExperiment->setText(
            QString::number(controller_->GetSizeOfExperimentDataset()));
        file_experiment_ = 1;
      }
    }
  } catch (std::out_of_range &exept) {
    ShowExeption(exept.what());
  }
}

void MainWindow::RunTestingClicked() {
  graph_->Clear();
  try {
    if (file_train_) {
      ui_->ProgressTraining->setValue(0);
      ui_->ProgressTrainingEpoch->setValue(0);

      controller_->SetMFunc(
          std::bind(&MainWindow::signalMetrics, this, std::placeholders::_1));
      controller_->SetPFunc(std::bind(&MainWindow::signalTrainProgress, this,
                                      std::placeholders::_1));
      controller_->SetFPFunc(std::bind(&MainWindow::signalFullTrainProgress,
                                       this, std::placeholders::_1));
      BlockButton(false);
      if (ui_->tabWidgetTraining->currentIndex() == 0) {
        graph_->SetRange(ui_->EpochNumber->currentText().toInt());
        std::thread trd([this]() {
          controller_->Train(s21::Config::TrainType::kTrain,
                             ui_->EpochNumber->currentText().toInt(),
                             ui_->LearningRate->value());
          BlockButton(true);
        });
        trd.detach();

      } else {
        graph_->SetRange(ui_->GroupsNumber->currentText().toInt());
        std::thread trd([this]() {
          controller_->Train(s21::Config::TrainType::kCrossValidation,
                             ui_->GroupsNumber->currentText().toInt(),
                             ui_->LearningRate->value());
          BlockButton(true);
        });
        trd.detach();
      }
    }
  } catch (std::out_of_range &exept) {
    ShowExeption(exept.what());
  }
}

void MainWindow::HiddenLayersChanged(const QString &arg1) {
  controller_->UpdateTopology(arg1.toInt());
  ui_->NumOfHiddenLayers->setText(arg1);
}

void MainWindow::RunExperimentClicked() {
  ClearExperimentLabel();
  try {
    if (file_experiment_) {
      controller_->SetMFunc(
          std::bind(&MainWindow::signalMetrics, this, std::placeholders::_1));
      controller_->SetPFunc(std::bind(&MainWindow::signalTestProgress, this,
                                      std::placeholders::_1));
      BlockButton(false);
      std::thread trd(

          [this]() {
            controller_->Test(ui_->SamplePart->value());
            BlockButton(true);
          });
      trd.detach();
    }
  } catch (std::out_of_range &exept) {
    ShowExeption(exept.what());
  }
}

void MainWindow::ExperimentOver(s21::Metrics metrics) {
  ui_->ProgressExperiment->setValue(0);
  ui_->AverageAccuracyResult->setText(
      QString::number(metrics.GetAccuracy() * 100.0, 'g', 4));
  ui_->PrecisionResult->setText(
      QString::number(metrics.GetPrecision() * 100.0, 'g', 4));
  ui_->RecallResult->setText(
      QString::number(metrics.GetRecall() * 100.0, 'g', 4));
  ui_->FMeasureResult->setText(
      QString::number(metrics.GetF1Score() * 100.0, 'g', 4));
  ui_->TotalTimeResult->setText(
      QString::number(metrics.GetTotalTime(), 'l', 2));
}

void MainWindow::ClearExperimentLabel() {
  ui_->AverageAccuracyResult->setText("...");
  ui_->PrecisionResult->setText("...");
  ui_->RecallResult->setText("...");
  ui_->FMeasureResult->setText("...");
  ui_->TotalTimeResult->setText("...");
}

void MainWindow::UpdateTestBar(int percent) {
  ui_->ProgressExperiment->setValue(percent);
}

void MainWindow::UpdateTrainBar(int percent) {
  ui_->ProgressTraining->setValue(percent);
}

void MainWindow::UpdateFullTrainBar(int percent) {
  ui_->ProgressTrainingEpoch->setValue(percent);
}

void MainWindow::BlockButton(bool state) {
  state_ = state;
  ui_->PerceptronTypeComboBox->setEnabled(state);
  ui_->LoadWeights->setEnabled(state);
  ui_->SaveWeights->setEnabled(state);
  ui_->LoadDataTraining->setEnabled(state);
  ui_->LoadDataExperiment->setEnabled(state);
  ui_->HiddenLayers->setEnabled(state);
  ui_->LearningRate->setEnabled(state);
  ui_->tabWidgetTraining->setEnabled(state);
  ui_->EpochNumber->setEnabled(state);
  ui_->GroupsNumber->setEnabled(state);
  ui_->tabWidget->setEnabled(state);
  ui_->SamplePart->setEnabled(state);
  ui_->RunTesting->setEnabled(state);
  ui_->RunExperiment->setEnabled(state);
  ui_->Result->setEnabled(state);
  ui_->LoadPicture->setEnabled(state);
}

void MainWindow::ShowExeption(QString exept) {
  QMessageBox msg_box;
  msg_box.setText(exept);
  msg_box.exec();
}

void MainWindow::closeEvent(QCloseEvent *event) {
  if (state_ == false) {
    event->ignore();
  } else {
    event->accept();
  }
}
