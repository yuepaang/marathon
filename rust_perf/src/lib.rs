/**
 * File              : lib.rs
 * Author            : Yue Peng <yuepaang@gmail.com>
 * Date              : 14.09.2023
 * Last Modified Date: 26.09.2023
 * Last Modified By  : Yue Peng <yuepaang@gmail.com>
 */
use std::{
    cmp::Ordering,
    collections::HashMap,
    collections::{HashSet, VecDeque},
    sync::{Arc, Mutex},
};

use algo::bfs;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::algo::check_out_of_bound;

// use crate::algo::{deal_with_enemy_nearby, move_enemy};

mod algo;
mod conf;

pub type Point = (i32, i32);

pub fn distance(p1: Point, p2: Point) -> i32 {
    (p1.0.abs_diff(p2.0) + p1.1.abs_diff(p2.1)) as i32
}

#[derive(Clone, Eq, PartialEq)]
pub struct Node {
    cost: i32,
    point: Point,
    passwall: usize,
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

#[pyfunction]
fn get_direction(start: Point, end: Point, blocked: Vec<Point>) -> PyResult<String> {
    let blocked: HashSet<Point> = blocked.into_iter().collect();
    let direction = algo::astar(start, end, blocked).unwrap_or("STAY".to_string());
    Ok(direction)
}

#[pyfunction]
fn get_direction_path(
    start: Point,
    end: Point,
    blocked: Vec<Point>,
) -> PyResult<(i32, Vec<String>)> {
    let blocked: HashSet<Point> = blocked.into_iter().collect();
    let direction_path = algo::astar_path(start, end, blocked).unwrap();
    Ok(direction_path)
}

fn compute_centroid(enemies: &Vec<Point>) -> Point {
    let mut sum_x = 0;
    let mut sum_y = 0;
    for &(x, y) in enemies.iter() {
        sum_x += x;
        sum_y += y;
    }
    (sum_x / enemies.len() as i32, sum_y / enemies.len() as i32)
}

fn get_diagonal_move(my_position: Point, centroid: Point) -> Point {
    let (my_x, my_y) = my_position;
    let (cent_x, cent_y) = centroid;

    // Determine the diagonal direction
    let possible_moves = [
        (my_x, my_y + 1), // up
        (my_x, my_y - 1), // down
        (my_x - 1, my_y), // left
        (my_x + 1, my_y), // right
    ];

    let mut best_move = my_position;
    let mut min_distance = f32::MAX;

    for &(move_x, move_y) in possible_moves.iter() {
        let distance = ((move_x - cent_x).pow(2) + (move_y - cent_y).pow(2)) as f32;
        if distance < min_distance {
            min_distance = distance;
            best_move = (move_x, move_y);
        }
    }

    best_move
}

fn get_escape_move(agent: Point, enemies: &Vec<Point>) -> Point {
    let (ax, ay) = agent;

    // Calculate the average position of enemies
    let (ex_avg, ey_avg) = enemies
        .iter()
        .fold((0, 0), |(sum_x, sum_y), &(x, y)| (sum_x + x, sum_y + y));
    let ex_avg = ex_avg / enemies.len() as i32;
    let ey_avg = ey_avg / enemies.len() as i32;

    // Choose a direction to move away from the average position of enemies
    if ax < ex_avg {
        return (agent.0 - 1, agent.1);
    } else if ax > ex_avg {
        return (agent.0 + 1, agent.1);
    } else if ay < ey_avg {
        return (agent.0, agent.1 - 1);
    } else {
        return (agent.0, agent.1 + 1);
    }
}

#[pyfunction]
fn check_stay_or_not(
    start: Point,
    mut enemies_position: Vec<Point>,
    pass_wall: usize,
    eaten_coins: HashSet<Point>,
) -> PyResult<Vec<Point>> {
    let banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();

    let mut enemies_all_pos = HashSet::new();
    for &e in enemies_position.iter() {
        for &(i, j) in &[(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)] {
            let next = (e.0 + i, e.1 + j);
            if algo::check_out_of_bound(next) {
                continue;
            }
            if banned_points.contains(&next) {
                continue;
            }
            enemies_all_pos.insert(next);
        }
    }

    let coin_set: HashSet<Point> = conf::COINS
        .par_iter()
        .chain(conf::POWERUPS.par_iter())
        .filter(|&c| !eaten_coins.contains(c))
        .map(|&c| (c.0, c.1))
        .collect();

    if enemies_position.len() == 2 {
        let centroid = compute_centroid(&enemies_position);
        let move_to = get_diagonal_move(start, centroid);
        let escape_to = get_escape_move(start, &enemies_position);

        if !banned_points.contains(&move_to)
            && !enemies_all_pos.contains(&move_to)
            && !check_out_of_bound(move_to)
        {
            return Ok(vec![move_to]);
        }
        if !banned_points.contains(&escape_to)
            && !enemies_all_pos.contains(&escape_to)
            && !check_out_of_bound(escape_to)
        {
            return Ok(vec![escape_to]);
        }
    }

    // Calculate the overall direction if the cell is further away from all enemies
    for &(i, j, consider_coin) in &[
        (-1, 0, true),
        (1, 0, true),
        (0, 1, true),
        (0, -1, true),
        (-1, 0, false),
        (1, 0, false),
        (0, 1, false),
        (0, -1, false),
    ] {
        let mut all_away = true;
        let mut next = (start.0 + i, start.1 + j);
        if algo::check_out_of_bound(next) {
            continue;
        }
        if consider_coin {
            if !coin_set.contains(&next) {
                continue;
            }
        }

        let start_index = conf::PORTALS
            .iter()
            .position(|portal| start.0 == portal.0 && start.1 == portal.1)
            .unwrap_or(99);
        let index = conf::PORTALS
            .iter()
            .position(|portal| next.0 == portal.0 && next.1 == portal.1)
            .unwrap_or(99);
        // DO NOT GO BACK after using portal
        if index != 99 && start_index != 99 {
            next = (conf::PORTALS_DEST[index].0, conf::PORTALS_DEST[index].1);
        }
        if enemies_all_pos.contains(&next) {
            continue;
        }

        if pass_wall == 0 {
            if banned_points.contains(&next) {
                continue;
            }
        } else {
            if banned_points.contains(&next) {
                return Ok(vec![next]);
            }
        }

        for enemy in &enemies_position {
            if shortest_path(start, *enemy).unwrap() > 5 {
                continue;
            }
            let new_path = shortest_path(next, *enemy).unwrap();
            let old_path = shortest_path(start, *enemy).unwrap();
            if new_path <= old_path {
                all_away = false;
            }
        }
        if all_away {
            return Ok(vec![next]);
        }
    }

    enemies_position.sort_by_key(|&e| shortest_path(start, e).unwrap());

    if enemies_position.len() == 1 && distance(start, enemies_position[0]) == 2 {
        return Ok(vec![start]);
    }
    if enemies_position.len() == 2 {
        if distance(start, enemies_position[0]) == 2 && distance(start, enemies_position[1]) > 2 {
            return Ok(vec![start]);
        }
    }

    let mut is_diag = true;
    for enemy in enemies_position.clone() {
        if (enemy.0 - start.0).abs() != 1 || (enemy.1 - start.1).abs() != 1 {
            is_diag = false;
            break;
        }
    }
    if is_diag {
        return Ok(vec![start]);
    }

    for &(i, j) in &[(-1, 0), (1, 0), (0, 1), (0, -1)] {
        let next = (start.0 + i, start.1 + j);
        if algo::check_out_of_bound(next) {
            continue;
        }
        if banned_points.contains(&next) {
            continue;
        }
        if shortest_path(next, enemies_position[0]).unwrap()
            > shortest_path(start, enemies_position[0]).unwrap()
        {
            return Ok(vec![next]);
        }
    }

    println!(
        "NO RUN AWAY! current: {:?} enemies: {:?}",
        start, enemies_position
    );

    // TODO: stay?
    Ok(vec![start])
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

#[pyfunction]
fn catch_enemies_using_powerup(
    start: Point,
    pass_wall: usize,
    mut enemies: Vec<Point>,
    defender_next_move: HashMap<Point, Point>,
    move_surround: bool,
) -> PyResult<Vec<Point>> {
    let banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    let chase_path;
    enemies.sort_by_key(|&e| shortest_path(start, e).unwrap());
    if !move_surround {
        println!("chase, start: {:?} enemies: {:?}", start, enemies);
        chase_path = algo::a_star_search_power(
            start,
            enemies[0],
            pass_wall,
            &banned_points,
            &HashSet::new(),
            &HashMap::new(),
        )
        .unwrap_or(vec![]);
    } else {
        let move_direction = defender_next_move.get(&enemies[0]).unwrap();
        println!(
            "surround, start: {:?} enemies: {:?}",
            start,
            (
                enemies[0].0 + move_direction.0 * 3,
                enemies[0].1 + move_direction.1 * 3,
            ),
        );
        chase_path = algo::a_star_search_power(
            start,
            (
                enemies[0].0 + move_direction.0 * 3,
                enemies[0].1 + move_direction.1 * 3,
            ),
            pass_wall,
            &banned_points,
            &HashSet::new(),
            &HashMap::new(),
        )
        .unwrap_or(vec![]);
    }
    return Ok(chase_path);
}

#[pyfunction]
fn collect_coins_using_hull(start: Point, eaten_coins: HashSet<Point>) -> PyResult<Vec<Point>> {
    let coins_vec: Vec<Point> = conf::COINS
        .par_iter()
        .map(|x| (x.0, x.1))
        .filter(|p| !eaten_coins.contains(p))
        .collect();

    let mut hull_points = algo::graham_hull(coins_vec);
    hull_points.sort_by_key(|&p| distance(start, p));
    let mut collect_path = Vec::new();

    let mut target_hull_point = start.clone();
    let mut target_dist = i32::MAX;
    for &hp in hull_points.iter() {
        let mut dist = distance(start, hp);

        let mut use_nearby = false;
        let mut new_hp = (hp.0, hp.1);
        if conf::WALLS.contains(&hp) {
            for &(i, j) in &[(-1, 0), (1, 0), (0, 1), (0, -1)] {
                new_hp = (hp.0 + i, hp.1 + j);
                if !conf::WALLS.contains(&new_hp) {
                    use_nearby = true;
                    break;
                }
            }
        }
        if use_nearby {
            dist = distance(start, new_hp);
        }
        if dist < target_dist {
            target_dist = dist;
            if !use_nearby {
                target_hull_point = hp.clone();
            } else {
                target_hull_point = new_hp.clone();
            }
        }
    }
    let hull_path = algo::a_star_search(start, target_hull_point).unwrap();
    collect_path.extend_from_slice(&hull_path);
    Ok(collect_path)
}

// BASELINE of defenders with some simple strategies
#[pyfunction]
fn collect_coins_using_powerup(
    agent_id: usize,
    mut start: Point,
    mut eaten_coins: HashSet<Point>,
    mut pass_wall: usize,
    enemies_position: HashSet<Point>,
    openess_map: HashMap<Point, i32>,
    max_depth: usize,
) -> PyResult<(Vec<Point>, Vec<Point>, usize)> {
    let origin = start.clone();
    // try not using portals
    let banned_points: HashSet<_> = conf::WALLS
        .par_iter()
        .chain(conf::PORTALS.par_iter())
        .cloned()
        .collect();
    let mut enemies_all_pos = HashSet::new();
    for &e in enemies_position.iter() {
        for &(i, j) in &[(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)] {
            let next = (e.0 + i, e.1 + j);
            if algo::check_out_of_bound(next) {
                continue;
            }
            if banned_points.contains(&next) {
                continue;
            }
            enemies_all_pos.insert(next);
        }
    }

    let mut search_depth = 0;
    let mut agent_coins_score = 0;
    let mut total_path = Vec::new();
    let mut first_coin_group = Vec::new();

    // find the potential move with greatest coin score
    loop {
        search_depth += 1;
        let positive_targets: Vec<Point> = conf::COINS
            .par_iter()
            .chain(conf::POWERUPS.par_iter())
            .filter(|x| !eaten_coins.contains(&x))
            .map(|x| (x.0, x.1))
            .collect();

        if search_depth > max_depth {
            break;
        }

        let paths: Vec<Vec<Point>> = positive_targets
            .par_iter()
            .filter_map(|&target| {
                algo::a_star_search_power(
                    start,
                    target,
                    pass_wall,
                    &banned_points,
                    &enemies_all_pos,
                    &openess_map,
                )
            })
            .collect();

        // there is no potential target
        if paths.is_empty() && search_depth == 1 {
            // TODO: avoid enemy within distance 1
            // println!(
            //     "agent {},NO TARGET! o: {:?} current position: {:?}, targets: {:?}, eaten number: {:?}, enemies: {:?}",
            //     agent_id,
            //     origin,
            //     start,
            //     positive_targets,
            //     eaten_coins.len(),
            //     enemies_position,
            // );
            return Ok((first_coin_group, total_path, agent_coins_score));
            // paths = conf::DEFENDER_BASE
            //     .par_iter()
            //     .chain(conf::ATTACKER_BASE.par_iter())
            //     .filter_map(|&p| {
            //         algo::a_star_search_power(
            //             start,
            //             p,
            //             pass_wall,
            //             &banned_points,
            //             &enemies_position,
            //         )
            //     })
            //     .collect();
        }

        // if paths.is_empty() && search_depth == 1 {
        //     for &e in enemies_position.iter() {
        //         if shortest_path(start, e).unwrap() > 2 {
        //             continue;
        //         }
        //         for &(i, j) in &[(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)] {
        //             let next = (origin.0 + i, origin.1 + j);
        //             if algo::check_out_of_bound(next) {
        //                 continue;
        //             }
        //             if shortest_path(next, e).unwrap() > shortest_path(origin, e).unwrap() {
        //                 total_path.push(next);
        //                 return Ok((first_coin_group, total_path, agent_coins_score));
        //             }
        //         }
        //     }
        //     println!("No strategy for now: JUST STAY.");
        //     println!(
        //         "agent {}, o:{:?}, current position: {:?}, targets: {:?}, eaten number: {:?}, enemies: {:?}",
        //         agent_id,
        //         origin,
        //         start,
        //         positive_targets,
        //         eaten_coins.len(),
        //         enemies_position,
        //     );
        // }

        if paths.is_empty() {
            break;
        }

        let sp = paths
            .iter()
            .filter(|path| path.len() > 0)
            .min_by_key(|path| path.len())
            .unwrap();

        total_path.extend_from_slice(&sp[..sp.len()]);

        start = *sp.last().unwrap();
        first_coin_group.push(start.clone());
        eaten_coins.insert((start.0, start.1));
        agent_coins_score += 2;
        pass_wall -= sp.len();
    }

    if total_path.len() == 0 {
        println!("no total path generated.");
        println!(
            "!!!!!!!!!agent {}, origin:{:?}, current position: {:?}, eaten number: {:?}, enemies: {:?}",
            agent_id,
            origin,
            start,
            eaten_coins.len(),
            enemies_position,
        );
    }

    Ok((first_coin_group, total_path, agent_coins_score))
}

// PERF: all agents parallel
#[pyfunction]
fn explore_n_round_scores(
    start: Point,
    real_eaten_coins: HashSet<Point>,
    enemies_position: Vec<Point>,
    pass_wall: usize,
) -> PyResult<Vec<f32>> {
    let mut banned_points: HashSet<_> = conf::WALLS.iter().cloned().collect();
    banned_points.insert((23, 0));

    // STAY, LEFT, RIGHT, DOWN, UP
    // let mut action_scores = vec![0.0, 0.0, 0.0, 0.0, 0.0];

    let eaten_coins = real_eaten_coins.clone();

    // let visited = HashSet::new();

    let search_depth = match enemies_position.len() {
        0 => 9,
        _ => 3,
    };

    let action_scores = bfs(
        start,
        search_depth,
        pass_wall,
        start,
        vec![],
        &eaten_coins,
        &enemies_position,
        &banned_points,
    );

    Ok(action_scores)
}

// #[pyfunction]
// fn explore(
//     agents: Vec<usize>,
//     positions: Vec<HashMap<usize, Point>>,
//     map_score: Vec<Vec<f32>>,
//     visited: HashMap<usize, HashSet<Point>>,
//     max_step: usize,
// ) -> PyResult<Vec<f32>> {
//     let result = Arc::new(Mutex::new(Vec::new()));
//     for _ in positions.iter() {
//         let mut result = result.lock().unwrap();
//         result.push(0.0);
//     }
//     println!("agents: {:?} pos_len: {}", agents, positions.len());
//     positions.par_iter().enumerate().for_each(|(idx, pos)| {
//         let mut max_reward = vec![0.0];

//         algo::dfs(
//             &agents,
//             &pos,
//             &map_score,
//             0.0,
//             &visited,
//             0,
//             max_step,
//             &mut max_reward,
//         );
//         let mut result = result.lock().unwrap();
//         result[idx] = max_reward[0];
//     });
//     Ok(Arc::try_unwrap(result).unwrap().into_inner().unwrap())
// }

#[pyfunction]
fn predict_enemy(
    agents: Vec<usize>,
    positions: Vec<HashMap<usize, Point>>,
    my_pos: HashMap<usize, Point>,
    visited: HashMap<usize, HashSet<Point>>,
    max_step: usize,
) -> PyResult<Vec<i32>> {
    let result = Arc::new(Mutex::new(Vec::new()));
    for _ in positions.iter() {
        let mut result = result.lock().unwrap();
        result.push(0);
    }
    // println!(
    //     "agents: {:?} pos_len: {}, my_pos: {:?}",
    //     agents,
    //     positions.len(),
    //     my_pos,
    // );
    positions.par_iter().enumerate().for_each(|(idx, pos)| {
        let mut max_reward = vec![0];

        algo::dfs(
            &agents,
            &pos,
            &my_pos,
            0,
            &visited,
            0,
            max_step,
            &mut max_reward,
        );
        let mut result = result.lock().unwrap();
        result[idx] = max_reward[0];
    });
    Ok(Arc::try_unwrap(result).unwrap().into_inner().unwrap())
}

#[pyfunction]
fn shortest_path(start: Point, end: Point) -> Option<i32> {
    let directions: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];
    let mut walls: HashSet<Point> = conf::WALLS.par_iter().map(|x| (x.0, x.1)).collect();
    walls.remove(&end);
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut portals = HashMap::new();
    for (idx, portal) in conf::PORTALS.iter().enumerate() {
        portals.insert(portal, conf::PORTALS_DEST[idx].clone());
    }
    queue.push_back((start, 0));

    while let Some((current, dist)) = queue.pop_front() {
        if current == end {
            return Some(dist);
        }

        // Check if current point is a portal entrance
        if let Some(&portal_exit) = portals.get(&current) {
            if !visited.contains(&portal_exit) {
                visited.insert(portal_exit);
                queue.push_back((portal_exit, dist + 1));
                continue;
            }
        }

        for &direction in directions.iter() {
            let next_point = (current.0 + direction.0, current.1 + direction.1);

            if !visited.contains(&next_point) {
                if walls.contains(&next_point) {
                    continue;
                }

                visited.insert(next_point);
                queue.push_back((next_point, dist + 1));
            }
        }
    }

    None
}

#[pymodule]
fn rust_perf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_direction, m)?)?;
    m.add_function(wrap_pyfunction!(get_direction_path, m)?)?;
    m.add_function(wrap_pyfunction!(collect_coins, m)?)?;
    m.add_function(wrap_pyfunction!(collect_coins_using_powerup, m)?)?;
    m.add_function(wrap_pyfunction!(catch_enemies_using_powerup, m)?)?;
    m.add_function(wrap_pyfunction!(check_stay_or_not, m)?)?;
    m.add_function(wrap_pyfunction!(explore_n_round_scores, m)?)?;
    m.add_function(wrap_pyfunction!(collect_coins_using_hull, m)?)?;
    m.add_function(wrap_pyfunction!(predict_enemy, m)?)?;
    m.add_function(wrap_pyfunction!(shortest_path, m)?)?;
    Ok(())
}
