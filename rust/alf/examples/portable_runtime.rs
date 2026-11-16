// Execute the same reader contracts natively and as 32-bit WebAssembly.
#[path = "../tests/counts.rs"]
mod counts;
#[path = "../tests/geometry_bounds.rs"]
mod geometry_bounds;

#[no_mangle]
pub extern "C" fn verify_counts() {
    counts::verify_counts();
}

#[no_mangle]
pub extern "C" fn verify_geometry_bounds() {
    geometry_bounds::verify_geometry_bounds();
}
