/**
 * File              : algo.rs
 * Author            : Yue Peng <yuepaang@gmail.com>
 * Date              : 21.09.2023
 * Last Modified Date: 21.09.2023
 * Last Modified By  : Yue Peng <yuepaang@gmail.com>
 */
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::conf;
use crate::Direction;
use crate::Node;
use crate::Point;

pub fn a_star_search(start: Point, end: Point) -> Option<Vec<Point>> {
    let banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();

    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from: HashMap<Point, (Direction, Point)> = HashMap::new();

    while let Some(Node { cost: _, point }) = to_visit.pop() {
        if point == end {
            let mut path: Vec<Point> = Vec::new();
            let mut current = end;

            while let Some(&(direction, parent)) = came_from.get(&current) {
                let next_pos: Point = direction.get_next_pos(parent);
                path.push(next_pos);
                current = parent;
            }

            path.reverse();
            return Some(path);
        }

        for &(i, j, direction) in &[
            (0, 0, Direction::Stay),
            (-1, 0, Direction::Left),
            (1, 0, Direction::Right),
            (0, 1, Direction::Down),
            (0, -1, Direction::Up),
        ] {
            let mut next = (point.0 + i, point.1 + j);
            if next.0 < 0 || next.0 >= conf::WIDTH || next.1 < 0 || next.1 >= conf::HEIGHT {
                continue;
            }
            if visited.contains(&next) || banned_points.contains(&next) {
                continue;
            }

            // PORTAL MOVE
            let index = conf::PORTALS
                .iter()
                .position(|portal| next.0 == portal.0 && next.1 == portal.1)
                .unwrap_or(99);
            if index != 99 {
                next = (conf::PORTALS_DEST[index].0, conf::PORTALS_DEST[index].1);
            }

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&usize::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score + crate::distance(next, end);
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                });
            }
        }

        visited.insert(point);
    }
    None
}

pub fn move_enemy(enemy_position: Point, target_position: Point) -> Point {
    if let Some(path) = a_star_search(enemy_position, target_position) {
        if path.len() >= 1 {
            return path[0];
        }
    }
    enemy_position.clone()
}

pub fn a_star_search_power(
    start: Point,
    end: Point,
    enemies: Vec<Point>,
    mut passwall: usize,
) -> Option<Vec<Point>> {
    let mut banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    banned_points.insert((23, 0));

    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from: HashMap<Point, (Direction, Point)> = HashMap::new();

    while let Some(Node { cost: _, point }) = to_visit.pop() {
        if point == end {
            let mut path: Vec<Point> = Vec::new();
            let mut current = end;

            while let Some(&(direction, parent)) = came_from.get(&current) {
                let next_pos: Point = direction.get_next_pos(parent);
                path.push(next_pos);
                current = parent;
            }

            path.reverse();
            return Some(path);
        }

        for &(i, j, direction) in &[
            (0, 0, Direction::Stay),
            (-1, 0, Direction::Left),
            (1, 0, Direction::Right),
            (0, 1, Direction::Down),
            (0, -1, Direction::Up),
        ] {
            let mut next = (point.0 + i, point.1 + j);
            if next.0 < 0 || next.0 >= conf::WIDTH || next.1 < 0 || next.1 >= conf::HEIGHT {
                continue;
            }
            if visited.contains(&next) {
                continue;
            }
            if enemies.contains(&next) {
                continue;
            }
            if passwall <= 0 {
                if banned_points.contains(&next) {
                    continue;
                }
            }

            // PORTAL MOVE
            let index = conf::PORTALS
                .iter()
                .position(|portal| next.0 == portal.0 && next.1 == portal.1)
                .unwrap_or(99);
            if index != 99 {
                next = (conf::PORTALS_DEST[index].0, conf::PORTALS_DEST[index].1);
            }

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&usize::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score + crate::distance(next, end);
                to_visit.push(Node {
                    cost: f_score,
                    point: next,
                });
            }
        }

        visited.insert(point);
        passwall -= 1;
    }
    None
}
