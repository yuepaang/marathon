/**
 * File              : lib.rs
 * Author            : Yue Peng <yuepaang@gmail.com>
 * Date              : 14.09.2023
 * Last Modified Date: 14.09.2023
 * Last Modified By  : Yue Peng <yuepaang@gmail.com>
 */
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
};

use pyo3::prelude::*;
use std::collections::HashMap;

mod algo;
mod conf;

pub type Point = (i32, i32);

pub fn distance(p1: Point, p2: Point) -> usize {
    (p1.0.abs_diff(p2.0) + p1.1.abs_diff(p2.1)) as usize
}

#[derive(Clone, Eq, PartialEq)]
pub struct Node {
    cost: usize,
    point: Point,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    Stay,
    Up,
    Down,
    Left,
    Right,
}

impl ToString for Direction {
    fn to_string(&self) -> String {
        match self {
            Direction::Stay => "STAY".to_string(),
            Direction::Up => "UP".to_string(),
            Direction::Down => "DOWN".to_string(),
            Direction::Left => "LEFT".to_string(),
            Direction::Right => "RIGHT".to_string(),
        }
    }
}
impl Direction {
    fn get_next_pos(&self, pos: Point) -> Point {
        match self {
            Direction::Stay => (pos.0, pos.1),
            Direction::Up => (pos.0, pos.1 + 1),
            Direction::Down => (pos.0, pos.1 - 1),
            Direction::Left => (pos.0 - 1, pos.1),
            Direction::Right => (pos.0 + 1, pos.1),
        }
    }
}

fn astar(start: Point, end: Point, blocked: HashSet<Point>) -> Option<String> {
    let mut banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    banned_points.extend(blocked);

    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from = HashMap::new();

    while let Some(Node { cost: _, point }) = to_visit.pop() {
        if point == end {
            let mut path: Vec<Direction> = Vec::new();
            let mut current = end;

            while let Some(&(direction, parent)) = came_from.get(&current) {
                path.push(direction);
                current = parent;
            }

            path.reverse();
            return Some(path.get(0).unwrap().to_string());
        }

        for &(i, j, direction) in &[
            (-1, 0, Direction::Left),
            (1, 0, Direction::Right),
            (0, -1, Direction::Down),
            (0, 1, Direction::Up),
        ] {
            let mut next = (point.0 + i, point.1 + j);

            if visited.contains(&next) || banned_points.contains(&next) {
                continue;
            }

            // PORTAL MOVE
            let index = conf::PORTALS
                .iter()
                .position(|portal| next.0 == portal.0 && next.1 == portal.1)
                .unwrap();

            next = (conf::PORTALS_DEST[index].0, conf::PORTALS_DEST[index].0);

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&usize::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score + distance(next, end);
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

fn astar_path(start: Point, end: Point, blocked: HashSet<Point>) -> Option<(usize, Vec<String>)> {
    let mut banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    banned_points.extend(blocked);

    let mut to_visit = BinaryHeap::new();
    to_visit.push(Node {
        cost: 0,
        point: start,
    });

    let mut visited = HashSet::new();
    let mut g_scores = HashMap::new();
    g_scores.insert(start, 0);

    let mut came_from = HashMap::new();

    while let Some(Node { cost, point }) = to_visit.pop() {
        if point == end {
            let mut path: Vec<Direction> = Vec::new();
            let mut current = end;

            while let Some(&(direction, parent)) = came_from.get(&current) {
                path.push(direction);
                current = parent;
            }

            path.reverse();
            let direction_path = path.into_iter().map(|x| x.to_string()).collect();
            return Some((cost, direction_path));
        }

        for &(i, j, direction) in &[
            (-1, 0, Direction::Left),
            (1, 0, Direction::Right),
            (0, -1, Direction::Down),
            (0, 1, Direction::Up),
        ] {
            let mut next = (point.0 + i, point.1 + j);

            if visited.contains(&next) || banned_points.contains(&next) {
                continue;
            }

            // PORTAL MOVE
            let index = conf::PORTALS
                .iter()
                .position(|portal| next.0 == portal.0 && next.1 == portal.1)
                .unwrap();

            next = (conf::PORTALS_DEST[index].0, conf::PORTALS_DEST[index].0);

            let tentative_g_score = g_scores.get(&point).unwrap() + 1;
            if tentative_g_score < *g_scores.get(&next).unwrap_or(&usize::MAX) {
                came_from.insert(next, (direction, point));
                g_scores.insert(next, tentative_g_score);
                let f_score = tentative_g_score + distance(next, end);
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

#[pyfunction]
fn get_direction(start: Point, end: Point, blocked: Vec<Point>) -> PyResult<String> {
    let blocked: HashSet<Point> = blocked.into_iter().collect();
    let direction = astar(start, end, blocked).unwrap_or("STAY".to_string());
    Ok(direction)
}

#[pyfunction]
fn get_direction_path(
    start: Point,
    end: Point,
    blocked: Vec<Point>,
) -> PyResult<(usize, Vec<String>)> {
    let blocked: HashSet<Point> = blocked.into_iter().collect();
    let direction_path = astar_path(start, end, blocked).unwrap();
    Ok(direction_path)
}

#[pyfunction]
fn collect_coins_with_enemy(
    mut start: Point,
    enemies_position: Vec<Point>,
    eatten_coins: HashSet<Point>,
) -> PyResult<(Vec<Point>, usize)> {
    let mut agent_coins_score = 0;
    let mut total_path = Vec::new();
    let mut coins_set: HashSet<Point> = conf::COINS
        .iter()
        .filter(|x| !eatten_coins.contains(x))
        .filter(|x| !enemies_position.contains(x))
        .map(|x| (x.0, x.1))
        .collect();
    let mut change_target = true;
    let mut sp = Vec::new();
    let mut search_depth = 0;
    // println!("coins' size: {}", coins_set.len());
    loop {
        // Pre-calculate ememy's next position
        let next_enemies = enemies_position
            .iter()
            .map(|&enemy| algo::move_enemy(enemy.clone(), start))
            .collect::<Vec<Point>>();

        search_depth += 1;
        let coins: Vec<Point> = coins_set.iter().cloned().collect();

        if coins.is_empty() || search_depth > 5 {
            // println!(
            //     "Start reach all coins!{}, ({},{})",
            //     coins.len(),
            //     start.0,
            //     start.1
            // );
            break;
        }
        if change_target {
            let paths: Vec<Vec<Point>> = coins
                .iter()
                .filter_map(|&coin| algo::a_star_search(start, coin, next_enemies.clone()))
                .collect();

            if paths.is_empty() {
                // println!("Couldn't reach all coins!");
                return Ok((vec![], 0));
            }
            sp = paths
                .iter()
                .filter(|path| path.len() > 0)
                .min_by_key(|path| path.len())
                .unwrap()
                .clone();

            // println!("sp size: {}, coin: {}", sp.len(), coins.len());
            // if sp.len() == 0 {
            //     for path in paths {
            //         println!("path: {:?}-{:?}", path[0], path[1]);
            //         println!("{}", path.len());
            //     }
            // }
            total_path.extend_from_slice(&sp[..sp.len() - 1]);
            change_target = false;
        }

        start = sp.pop().unwrap();
        if coins_set.contains(&start) {
            // println!("Catch coin! Remaining coins: {}", agent_coins_score);
            coins_set.remove(&start);
            agent_coins_score += 2;
            change_target = true;
        }
        // println!("The next point is ({}, {})", start.0, start.1);
    }

    Ok((total_path, agent_coins_score))
}

#[pymodule]
fn rust_perf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_direction, m)?)?;
    m.add_function(wrap_pyfunction!(get_direction_path, m)?)?;
    m.add_function(wrap_pyfunction!(collect_coins_with_enemy, m)?)?;
    Ok(())
}
