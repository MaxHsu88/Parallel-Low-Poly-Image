#include "point.h"

Point::Point(int x_, int y_)
{
    x = x_;
    y = y_;
}

// bool Point::isInvalid()
// {
//     return x == -1 && y == -1;
// }

Point operator + (const Point &a, const Point &b)
{
    return Point(a.x + b.x, a.y + b.y);
}

Point operator * (const Point &a, int b)
{
    return Point(a.x * b, a.y * b);
}

Point operator / (const Point &a, int b)
{
    return Point(a.x / b, a.y / b);
}

bool operator == (const Point &a, const Point &b)
{
    return a.x == b.x && a.y == b.y;
}

int distance(const Point &a, const Point &b) 
{
    int dx = a.x - b.x;
    int dy = a.y - b.y;
    return dx * dx + dy * dy;
}