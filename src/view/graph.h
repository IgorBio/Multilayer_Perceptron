#ifndef GRAPH_H
#define GRAPH_H

#include <QChart>
#include <QChartView>
#include <QLineSeries>
#include <QValueAxis>
#include <QWidget>
#include <QtCharts>

#include "metrics.h"

class Graph : public QWidget {
  Q_OBJECT
 public:
  explicit Graph(QLayout* layout);
  ~Graph() {
    delete axis_x_;
    delete axis_y_;
    delete mse_;
    delete chart_;
    delete chart_view_;
  }

  void Draw(s21::Metrics mtrx);
  void SetRange(int epochs);
  void Clear();

 private:
  QValueAxis* axis_x_;
  QValueAxis* axis_y_;
  QLineSeries* mse_;
  QChart* chart_;
  QChartView* chart_view_;

  int count_epoch_ = 0;
};

#endif  // GRAPH_H
