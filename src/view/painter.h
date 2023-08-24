#ifndef PAINTER_H
#define PAINTER_H

#include <QImage>
#include <QMouseEvent>
#include <QPainter>
#include <QWidget>

#include "mainwindow.h"

class Painter : public QWidget {
  Q_OBJECT
 public:
  explicit Painter(QWidget* parent = nullptr);

  void SetWindow(MainWindow* mainWindow) { window_ = mainWindow; }

 private:
  QPoint mouse_pos_;
  QImage picture_of_letter_;
  MainWindow* window_ = nullptr;
  Qt::MouseButton mouse_but_;

  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void paintEvent(QPaintEvent* event) override;

  void Draw(const QPoint& move);
  bool CheckArea(int x, int y);
};

#endif  // PAINTER_H
