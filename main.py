from src import graduate_design


if __name__ == "__main__":
    graduate_design.load_model('automap')
    # graduate_design.train()
    graduate_design.validate()
    # graduate_design.predict('')
