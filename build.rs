use anyhow::*;
use fs_extra::{copy_items, dir::CopyOptions};
use std::env;

fn main() -> Result<()> {
    // Tell cargo to re-run the script if something in /res changes.
    println!("cargo:rerun-if-changed=res/*");

    let out_dir = env::var("OUT_DIR")?;
    let mut copy_options = CopyOptions::new();
    copy_options.overwrite = true;
    let mut paths_to_copy = Vec::new();
    paths_to_copy.push("res/");
    copy_items(&paths_to_copy, out_dir, &copy_options)?;

    Ok(())
}