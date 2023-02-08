from project import data


# def test_data_init():
#     data.init(100, 140, 3, 10, 30, 180 / 140, 0, 180)


def test_dataset():
    dataset = data.Dataset()
    with dataset:
        assert dataset[1] == dataset[1]
        assert dataset[33] == dataset[33]
        assert dataset[55][0].shape == (1, 140, 140)
