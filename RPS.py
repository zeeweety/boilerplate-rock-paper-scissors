# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

from sklearn.naive_bayes import MultinomialNB as model

clf = model()
classes = [1, 2, 3]
actions = ["R", "P", "S"]
optimum_action = {"R": "P", "P": "S", "S": "R"}
class_num = {"R": 1, "P": 2, "S": 3}
my_act_combo_stats = {}
for p2 in actions:
    for p1 in actions:
        my_act_combo_stats[(p2, p1)] = 0
opponent_history = ["S"]
my_history = ["R", "R"]


def player(prev_play):
    """Single strategy for all four players."""
    global clf, my_act_combo_stats, opponent_history, my_history

    if prev_play == "":
        prev_play = "S"
    opponent_history.append(prev_play)

    # Reset when start playing with different player
    if len(opponent_history) % 1000 == 2:
        opponent_history = ["S", "S"]
        my_history = ["R", "R"]
        clf = model()
        my_act_combo_stats = {}
        for p_2 in actions:
            for p_1 in actions:
                my_act_combo_stats[(p_2, p_1)] = 0

    p2, p1 = my_history[-2:]

    # Incrementally/online learning model with feature vector
    #  and output label from last round
    prior_plays = my_history[-11:-1]
    prior_actions_count = [0, 0, 0]
    for a in prior_plays:
        prior_actions_count[class_num[a] - 1] += 1
    oppo_second_last_action = opponent_history[-2]
    my_second_last_action = my_history[-2]
    train_input = [x for x in prior_actions_count]
    oppo_second_last_action_class_num = class_num[oppo_second_last_action]
    for i in range(1, 4):
        if i == oppo_second_last_action_class_num:
            train_input.append(1)
        else:
            train_input.append(0)
    my_second_last_action_class_num = class_num[my_second_last_action]
    for i in range(1, 4):
        if i == my_second_last_action_class_num:
            train_input.append(1)
        else:
            train_input.append(0)

    prev_expected_next_action = "P"
    prev_n_plays = 0
    for combo, count in my_act_combo_stats.items():
        if combo[0] == p2 and count > prev_n_plays:
            prev_expected_next_action = combo[1]
            prev_n_plays = count
    prev_expected_next_action_class_num = class_num[prev_expected_next_action]
    for i in range(1, 4):
        if i == prev_expected_next_action_class_num:
            train_input.append(1)
        else:
            train_input.append(0)

    clf.partial_fit(
        [train_input],
        [class_num[optimum_action[prev_play]]],
        classes)

    # Predicting action for this round
    my_act_combo_stats[(p2, p1)] += 1

    recent_plays = my_history[-10:]
    recent_actions_count = [0, 0, 0]
    for a in recent_plays:
        recent_actions_count[class_num[a] - 1] += 1
    my_prev_action = my_history[-1]
    predict_input = [x for x in recent_actions_count]
    oppo_last_action_class_num = class_num[prev_play]
    for i in range(1, 4):
        if i == oppo_last_action_class_num:
            predict_input.append(1)
        else:
            predict_input.append(0)
    my_last_action_class_num = class_num[my_prev_action]
    for i in range(1, 4):
        if i == my_last_action_class_num:
            predict_input.append(1)
        else:
            predict_input.append(0)

    expected_next_action = ""
    n_plays = 0
    for combo, count in my_act_combo_stats.items():
        if combo[0] == p1 and count >= n_plays:
            expected_next_action = combo[1]
            n_plays = count
    expected_next_action_class_num = class_num[expected_next_action]
    for i in range(1, 4):
        if i == expected_next_action_class_num:
            predict_input.append(1)
        else:
            predict_input.append(0)

    action = actions[clf.predict([predict_input])[0] - 1]

    my_history.append(action)

    return action
