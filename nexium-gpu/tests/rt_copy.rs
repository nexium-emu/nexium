use ash::vk;
use nexium_gpu::{rt_cache::RtKey, Renderer};

#[test]
#[ignore = "requires a Vulkan device"]
fn cropped_hdr_copy_preserves_bits_and_destination_pixels() {
    let renderer = Renderer::new().unwrap();
    let hdr = vk::Format::B10G11R11_UFLOAT_PACK32;
    let src = RtKey::new(1, 8, 8, 0x10000);
    renderer.clear_target_with_format(1, 8, 8, 0x10000, [4.0, 2.0, 0.5, 1.0], hdr).unwrap();
    let (_, _, bpp, original) = renderer.readback_target_raw_key(src).unwrap();
    assert_eq!(bpp, 4);
    assert_ne!(&original[..4], &[255, 255, 128, 255]);
    for (id, format) in [(2, hdr), (3, vk::Format::R32_UINT)] {
        let dst = RtKey::new(id, 4, 4, u64::from(id) * 0x10000);
        renderer.clear_target_with_format(id, 4, 4, dst.gpu_va, [0.0; 4], hdr).unwrap();
        assert!(renderer.resolve_rt_copy(1, 8, 8, 0x10000, dst,
            [2, 2, 4, 4], [1, 1, 3, 3], format, format).unwrap());
        let (_, _, copied_bpp, copied) = renderer.readback_target_raw_key(dst).unwrap();
        assert_eq!(copied_bpp, 4);
        for y in 0..4 {
            for x in 0..4 {
                let expected = if (1..3).contains(&x) && (1..3).contains(&y) {
                    &original[..4]
                } else {
                    &[0; 4]
                };
                assert_eq!(&copied[(y * 4 + x) * 4..(y * 4 + x + 1) * 4], expected);
            }
        }
    }
    let converted = RtKey::new(4, 2, 2, 0x40000);
    assert!(renderer.resolve_rt_copy(1, 8, 8, 0x10000, converted,
        [2, 2, 4, 4], [0, 0, 2, 2], hdr, vk::Format::R8G8B8A8_UNORM).unwrap());
    let (_, _, _, pixels) = renderer.readback_target_raw_key(converted).unwrap();
    assert_eq!(&pixels[..2], &[255, 255]);
    assert!((127..=128).contains(&pixels[2]));
    assert_eq!(pixels[3], 255);
}
