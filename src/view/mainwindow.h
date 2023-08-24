#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QFileDialog>
#include <QImageReader>
#include <QLabel>
#include <QMainWindow>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QVector>
#include <QWidget>
#include <functional>

#include "QtWidgets/qprogressbar.h"
#include "controller.h"
#include "graph.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

  void NormalizePicture(QImage picture);
  bool flag_load_picture_ = 0;
  void ClearResultingLetter();

  QLabel *label_ = nullptr;

 private slots:
  void LoadWeightsClicked();
  void LoadPictureClicked();
  void SaveWeightsClicked();
  void LoadDataTrainingClicked();
  void LoadDataExperimentClicked();
  void PerceptronTypeChanged(int index);
  void HiddenLayersChanged(const QString &arg1);
  void RunTestingClicked();
  void RunExperimentClicked();
  void PredictClicked();

  void ExperimentOver(s21::Metrics metrics);
  void UpdateTestBar(int percent);
  void UpdateTrainBar(int percent);
  void UpdateFullTrainBar(int percent);

 private:
  Ui::MainWindow *ui_;
  std::unique_ptr<s21::Controller> controller_ = nullptr;
  std::vector<double> vector_with_pix_;
  Graph *graph_;
  bool file_experiment_ = 0, file_train_ = 0;
  bool state_ = true;

  void ConnectSignals();
  void ShowExeption(QString exept);
  void LoadFile(bool is_train);
  void ClearExperimentLabel();
  void BlockButton(bool state);
  void closeEvent(QCloseEvent *event) override;

 signals:
  void signalMetrics(s21::Metrics metrics);
  void signalTrainProgress(int);
  void signalFullTrainProgress(int);
  void signalTestProgress(int);
};

#endif  // MAINWINDOW_H
