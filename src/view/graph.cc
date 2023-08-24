#include "graph.h"

#include "QtWidgets/qlayout.h"
#include "metrics.h"

Graph::Graph(QLayout* layout) {
  mse_ = new QLineSeries();
  chart_ = new QChart();
  chart_->addSeries(mse_);
  chart_->setTitle("MSE");
  chart_->legend()->hide();

  chart_view_ = new QChartView(chart_);
  layout->addWidget(chart_view_);

  axis_x_ = new QValueAxis;
  axis_x_->setRange(0, 1);
  axis_x_->setLabelFormat("%.0d");
  chart_->addAxis(axis_x_, Qt::AlignBottom);

  axis_y_ = new QValueAxis;
  axis_y_->setRange(0, 1);
  chart_->addAxis(axis_y_, Qt::AlignLeft);

  mse_->attachAxis(axis_x_);
  mse_->attachAxis(axis_y_);

  chart_view_->update();
}

void Graph::SetRange(int epochs) {
  axis_x_->setRange(0, epochs);
  axis_x_->setTickCount(epochs + 1);

  chart_view_->update();
}

void Graph::Draw(s21::Metrics mtrx) {
  count_epoch_++;
  mse_->append(count_epoch_, mtrx.GetLoss());
  chart_view_->update();
}

void Graph::Clear() {
  count_epoch_ = 0;
  mse_->clear();
  mse_->append(0, 1);
  chart_view_->update();
}
