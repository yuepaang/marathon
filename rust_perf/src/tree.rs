/**
 * File              : tree.rs
 * Author            : Yue Peng <yuepaang@gmail.com>
 * Date              : 26.09.2023
 * Last Modified Date: 26.09.2023
 * Last Modified By  : Yue Peng <yuepaang@gmail.com>
 */
enum Status {
    Success,
    Failure,
    Running,
}

trait TreeNode {
    fn tick(&self) -> Status;
}

struct Selector {
    children: Vec<Box<dyn TreeNode>>,
}

impl TreeNode for Selector {
    fn tick(&self) -> Status {
        for child in &self.children {
            match child.tick() {
                Status::Success => return Status::Success,
                Status::Running => return Status::Running,
                _ => (),
            }
        }
        Status::Failure
    }
}

struct Sequence {
    children: Vec<Box<dyn Node>>,
}

impl TreeNode for Sequence {
    fn tick(&self) -> Status {
        for child in &self.children {
            match child.tick() {
                Status::Failure => return Status::Failure,
                Status::Running => return Status::Running,
                _ => (),
            }
        }
        Status::Success
    }
}

struct Action;

impl TreeNode for Action {
    fn tick(&self) -> Status {
        // Place your action code here
        // Return the status based on the action outcome
        Status::Success
    }
}
