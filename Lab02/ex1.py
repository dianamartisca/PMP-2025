import random

def dice_action(number, urn_dictionary):
    if number==6:
        urn_dictionary["red"] += 1
    elif number in {2,3,5}:
        urn_dictionary["black"] += 1
    else:
        urn_dictionary["blue"] += 1

def simulation():
    urn_dictionary = {
    "red": 3,
    "blue": 4,
    "black": 2
    }
    dice_action(random.randint(1,6), urn_dictionary)
    balls = []
    for color, count in urn_dictionary.items():
        balls.extend([color] * count)
    drawn_ball = random.choice(balls)
    return drawn_ball

print(f"Drawn ball: {simulation()}")

def red_ball_probability(trials=10000):
    red_count = sum(1 for _ in range(trials) if simulation() == 'red')
    return red_count / trials

print(f"Estimated P(red) over 10000 trials: {red_ball_probability()}")

def theoretical_red_probability():
    red_ball_probability=3/10*1/2 + 4/10*1/6 + 3/10*1/3
    return red_ball_probability

print(f"Theoretical P(red): {theoretical_red_probability()}")
