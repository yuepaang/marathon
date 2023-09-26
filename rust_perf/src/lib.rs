/**
 * File              : lib.rs
 * Author            : Yue Peng <yuepaang@gmail.com>
 * Date              : 14.09.2023
 * Last Modified Date: 26.09.2023
 * Last Modified By  : Yue Peng <yuepaang@gmail.com>
 */
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
};

use algo::bfs;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::algo::deal_with_enemy_nearby;

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
            Direction::Up => (pos.0, pos.1 - 1),
            Direction::Down => (pos.0, pos.1 + 1),
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
            (0, 1, Direction::Down),
            (0, -1, Direction::Up),
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
            (0, 1, Direction::Down),
            (0, -1, Direction::Up),
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
fn check_two_enemies_move(start: Point, enemies_position: Vec<Point>) -> PyResult<String> {
    let mut is_diag = true;
    for enemy in enemies_position {
        if (enemy.0 - start.0).abs() != 1 || (enemy.1 - start.1).abs() != 1 {
            is_diag = false;
            break;
        }
    }
    if is_diag {
        Ok("STAY".to_string())
    } else {
        Ok("NO".to_string())
    }
}

// BASELINE 170 round end the game
#[pyfunction]
fn collect_coins(
    mut start: Point,
    mut eaten_coins: HashSet<Point>,
) -> PyResult<(Vec<Point>, usize)> {
    let mut agent_coins_score = 0;
    let mut total_path = Vec::new();
    let mut search_coins = 0;
    let depth = 9;
    loop {
        // Pre-calculate ememy's next position
        search_coins += 1;
        let coins: Vec<Point> = conf::COINS
            .par_iter()
            .filter(|x| !eaten_coins.contains(&x))
            .map(|x| (x.0, x.1))
            .collect();

        if coins.is_empty() || search_coins > depth {
            break;
        }

        let paths: Vec<Vec<Point>> = coins
            .iter()
            .filter_map(|&coin| algo::a_star_search(start, coin))
            .collect();

        if paths.is_empty() {
            return Ok((vec![], 0));
        }

        let sp = paths.iter().min_by_key(|path| path.len()).unwrap();
        total_path.extend_from_slice(&sp[..sp.len()]);

        start = *sp.last().unwrap();
        eaten_coins.insert((start.0, start.1));
        agent_coins_score += 2;
    }

    Ok((total_path, agent_coins_score))
}

// BASELINE of defenders with some simple strategies
#[pyfunction]
fn collect_coins_using_powerup(
    step: i32,
    agent_id: i32,
    mut start: Point,
    mut eaten_coins: HashSet<Point>,
    allies_position: Vec<Point>,
    enemies_position: Vec<Point>,
    mut pass_wall: usize,
) -> PyResult<(Vec<Point>, usize)> {
    let origin = start.clone();
    // let quad_id: i32 = agent_id - 4;

    let mut search_depth = 0;
    let mut agent_coins_score = 0;
    let mut total_path = Vec::new();
    let mut no_coin_situation = false;

    // let coins_vec: Vec<Point> = conf::COINS.iter().map(|x| (x.0, x.1)).collect();
    // let hull_points = algo::graham_hull(coins_vec);
    // println!("hull points: {:?}", hull_points);

    // check escape strategy
    let escape_path = deal_with_enemy_nearby(start, enemies_position.clone());
    if !escape_path.is_empty() {
        return Ok((escape_path, 0));
    }

    // safe and scatter phrase
    // if step < 30 {
    //     let can_collect_path = conf::COINS
    //         .par_iter()
    //         .chain(conf::POWERUPS.par_iter())
    //         .filter(|(x, y)| {
    //             let mut flag = false;
    //             match quad_id {
    //                 0 => flag = x < &12 && y < &12,
    //                 1 => flag = x < &12 && y >= &12,
    //                 2 => flag = x >= &12 && y >= &12,
    //                 3 => flag = x >= &12 && y < &12,
    //                 _ => (),
    //             };
    //             flag
    //         })
    //         .filter(|x| !eaten_coins.contains(&x))
    //         // .filter(|&x| distance(start, x.clone()) < 5)
    //         .filter_map(|&coin| algo::a_star_search(start, coin))
    //         .collect::<Vec<Vec<Point>>>();

    //     if !can_collect_path.is_empty() {
    //         let sp = can_collect_path
    //             .iter()
    //             .min_by_key(|path| path.len())
    //             .unwrap();

    //         // println!(
    //         //     "agent{}: ({},{}) can collect path: {:?}",
    //         //     agent_id, start.0, start.1, sp
    //         // );
    //         return Ok((sp.clone(), 0));
    //     }
    // }

    // find the potential move with greatest coin score

    loop {
        // Pre-calculate ememys' next position (assume chasing if in their vision)
        // let next_enemies = enemies_position
        //     .iter()
        //     .map(|&enemy| algo::move_enemy(enemy.clone(), start))
        //     .collect::<Vec<Point>>();

        search_depth += 1;
        let coins: Vec<Point> = conf::COINS
            .par_iter()
            .chain(conf::POWERUPS.par_iter())
            .filter(|x| !eaten_coins.contains(&x))
            .map(|x| (x.0, x.1))
            .collect();

        if coins.is_empty() || search_depth > 10 {
            break;
        }

        let mut paths: Vec<Vec<Point>> = coins
            .par_iter()
            .filter_map(|&coin| {
                algo::a_star_search_power(start, coin, allies_position.clone(), pass_wall)
            })
            .collect();

        if paths.is_empty() {
            println!("NO COINS CATCHED");
            paths = conf::PORTALS
                .par_iter()
                .filter_map(|&portal| {
                    algo::a_star_search_power(start, portal, allies_position.clone(), pass_wall)
                })
                .collect();
            no_coin_situation = true;
        }

        if no_coin_situation && paths.is_empty() {
            println!("No strategy for now: JUST STAY");
            total_path.push(origin);
            break;
        }

        let empty_path = vec![];
        let mut sp = paths
            .iter()
            .min_by_key(|path| path.len())
            .unwrap_or(&empty_path);

        if sp.len() == 0 {
            sp = paths.iter().min_by_key(|path| path.len()).unwrap();
        }
        total_path.extend_from_slice(&sp[..sp.len()]);

        if no_coin_situation {
            break;
        }

        start = *sp.last().unwrap();
        eaten_coins.insert((start.0, start.1));
        agent_coins_score += 2;
        pass_wall -= sp.len();
    }

    if total_path.len() == 0 {
        println!("no total path generated.")
    }

    Ok((total_path, agent_coins_score))
}

// PERF: all agents parallel
#[pyfunction]
fn explore_n_round_scores(
    start: Point,
    real_eaten_coins: HashSet<Point>,
    enemies_position: Vec<Point>,
    pass_wall: usize,
    // search_depth: usize,
) -> PyResult<Vec<f32>> {
    let mut banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    banned_points.insert((23, 0));

    // STAY, LEFT, RIGHT, DOWN, UP
    let mut action_scores = vec![0.0, 0.0, 0.0, 0.0, 0.0];

    let mut eaten_coins = real_eaten_coins.clone();

    let visited = HashSet::new();

    // dfs(
    //     start,
    //     0,
    //     pass_wall,
    //     start,
    //     vec![],
    //     &mut eaten_coins,
    //     &enemies_position,
    //     &banned_points,
    //     &mut action_scores,
    //     -1,
    // );
    let search_depth = match enemies_position.len() {
        0 => 9,
        _ => 3,
    };

    bfs(
        start,
        search_depth,
        pass_wall,
        start,
        vec![],
        &mut eaten_coins,
        &enemies_position,
        &banned_points,
        &mut action_scores,
        visited,
    );

    Ok(action_scores)
}

#[pymodule]
fn rust_perf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_direction, m)?)?;
    m.add_function(wrap_pyfunction!(get_direction_path, m)?)?;
    m.add_function(wrap_pyfunction!(collect_coins, m)?)?;
    m.add_function(wrap_pyfunction!(collect_coins_using_powerup, m)?)?;
    m.add_function(wrap_pyfunction!(check_two_enemies_move, m)?)?;
    m.add_function(wrap_pyfunction!(explore_n_round_scores, m)?)?;
    Ok(())
}
