import csv

from PIL import Image

from skinlesion.dedup import (
    dhash,
    dhash_file,
    find_cross_duplicates,
    hamming,
    sha256_file,
    write_report,
)
from tests.conftest import fake_image


def test_same_image_has_equal_sha256_and_dhash(tmp_path):
    img = fake_image(1, 0)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    img.save(a)
    img.save(b)
    assert sha256_file(a) == sha256_file(b)
    assert dhash_file(a) == dhash_file(b)


def test_resized_copy_is_within_hamming_4():
    img = fake_image(2, 1, size=96)
    small = img.resize((60, 60), Image.Resampling.BILINEAR)
    assert hamming(dhash(img), dhash(small)) <= 4


def test_different_images_are_far_apart():
    assert hamming(dhash(fake_image(3, 0)), dhash(fake_image(4, 1))) > 4


def test_dhash_is_64_bit():
    assert 0 <= dhash(fake_image(5, 0)) < 2**64


def test_cross_duplicates_finds_planted_pair_and_writes_csv(tmp_path):
    train = []
    for i in range(4):
        path = tmp_path / f"train_{i}.png"
        fake_image(10 + i, i % 2).save(path)
        train.append(str(path))
    planted = tmp_path / "test_copy.png"
    Image.open(train[2]).save(planted)
    unique = tmp_path / "test_unique.png"
    fake_image(99, 0).save(unique)

    pairs = find_cross_duplicates(train, [str(planted), str(unique)])

    assert [(p.train_path, p.test_path, p.exact) for p in pairs] == [(train[2], str(planted), True)]
    out = write_report(pairs, tmp_path / "reports" / "dedup.csv")
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["hamming"] == "0" and rows[0]["exact"] == "1"
