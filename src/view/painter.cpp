#include "painter.h"

#include "QtWidgets/qlabel.h"

Painter::Painter(QWidget* parent)
    : QWidget{parent},
      picture_of_letter_(QImage(QSize(512, 512), QImage::Format_RGB16)) {
  picture_of_letter_.fill(qRgb(255, 255, 255));
  update();
}

void Painter::mousePressEvent(QMouseEvent* event) {
  mouse_but_ = event->button();
  if (mouse_but_ == Qt::LeftButton && !window_->flag_load_picture_) {
    mouse_pos_ = event->pos();
  } else if (event->button() == Qt::RightButton) {
    picture_of_letter_.fill(qRgb(255, 255, 255));
    update();
    window_->ClearResultingLetter();
    if (window_->flag_load_picture_) {
      window_->label_->close();
      window_->flag_load_picture_ = 0;
    }
  }
}

void Painter::mouseMoveEvent(QMouseEvent* event) {
  if (mouse_but_ == Qt::LeftButton &&
      CheckArea((mouse_pos_.x()), mouse_pos_.y()) &&
      !window_->flag_load_picture_) {
    Draw(event->pos());
  }
}

void Painter::Draw(const QPoint& move) {
  QPainter line(&picture_of_letter_);
  line.setRenderHint(QPainter::SmoothPixmapTransform, true);
  line.setRenderHint(QPainter::Antialiasing, true);
  line.setPen(QPen(Qt::black, 30, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  line.drawLine(mouse_pos_, move);

  update();
  mouse_pos_ = move;
}

void Painter::mouseReleaseEvent(QMouseEvent* event) {
  if (!window_->flag_load_picture_) {
    window_->NormalizePicture(picture_of_letter_);
  }
}

bool Painter::CheckArea(int x, int y) {
  return (x >= 0 && x <= 512 && y >= 0 && y <= 512);
}

void Painter::paintEvent(QPaintEvent* event) {
  Q_UNUSED(event);
  QPainter eventPainter(this);
  eventPainter.drawImage(picture_of_letter_.rect(), picture_of_letter_);
}
