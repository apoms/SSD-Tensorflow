# Vgg like model to train and test on cifar-10 and cifar-100

import time
import six
import sys

import cifar_input
import numpy as np
import branch_resnet_model
import tensorflow as tf

import utils
import time
import heapq

def get_predictions(batch_size, num_classes, num_branches, num_stem_units,
                    branch_base_units, uniform_cost, exploit_ratio,
                    flop_ratio, weight_decay, lrn_rate, optimizer,
                    dataset, eval_data_path, eval_batch_count,
                    eval_dir, log_root, checkpoint_path = None,
                    model_type = 'split_task'):
  """Eval loop."""
  images, labels = cifar_input.build_input(dataset, eval_data_path, batch_size, 'eval')
  model = branch_resnet_model.BranchResNet(images, labels, batch_size,
                                           num_classes, exploit_ratio, flop_ratio,
                                           num_branches, uniform_cost,
                                           num_stem_units, branch_base_units,
                                           'eval', weight_decay,
                                           lrn_rate, optimizer,
                                           model_type = model_type)
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(eval_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision = 0.0

  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    if checkpoint_path:
        saver.restore(sess, checkpoint_path)
    else:
        saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0

    high_correct_prediction, high_total_prediction = 0, 0
    low_correct_prediction, low_total_prediction = 0, 0

    hard_total_prediction, hard_correct_prediction = 0, 0

    prediction_list = []
    prediction_confidence_list = []
    switch_confidence_list = []

    image_list = []

    for batch_num in six.moves.range(eval_batch_count):
      (summaries, loss, predictions, hard_predictions,
       branch_predictions, switch_predictions,
       truth, curr_images, train_step) = sess.run(
          [model.summaries, model.cost, model.predictions,
           model.hard_predictions, model.branch_predictions,
           model.switch_predictions, model.labels,
           model.images, model.global_step])

      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

      hard_confs = np.max(hard_predictions, axis=1)
      hard_predictions = np.argmax(hard_predictions, axis=1)
      hard_correct_prediction += np.sum(truth == hard_predictions)

      hard_total_prediction += hard_predictions.shape[0]

      switch_confs = np.max(switch_predictions, axis=1)

      for b in range(0, batch_size):
          image_list.append(curr_images[b])
          prediction_list.append(truth[b]== hard_predictions[b])
          prediction_confidence_list.append(hard_confs[b])
          switch_confidence_list.append(switch_confs[b])

    precision = 1.0 * correct_prediction / total_prediction
    hard_precision = 1.0 * hard_correct_prediction / hard_total_prediction
    break

  return (image_list, prediction_list,
          prediction_confidence_list, switch_confidence_list)

def analyze_failures(batch_size, num_classes, num_branches, num_stem_units,
                     branch_base_units, uniform_cost, exploit_ratio,
                     flop_ratio, weight_decay, lrn_rate, optimizer,
                     dataset, eval_data_path, eval_batch_count,
                     eval_dir, log_root, checkpoint_path = None,
                     model_type = 'split_task',
                     switch_confidence_thresh = 0.9):

  """Eval loop."""
  images, labels = cifar_input.build_input(dataset, eval_data_path, batch_size, 'eval')
  model = branch_resnet_model.BranchResNet(images, labels, batch_size,
                                           num_classes, exploit_ratio, flop_ratio,
                                           num_branches, uniform_cost,
                                           num_stem_units, branch_base_units,
                                           'eval', weight_decay,
                                           lrn_rate, optimizer,
                                           model_type = model_type)
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(eval_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision = 0.0

  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    if checkpoint_path:
        saver.restore(sess, checkpoint_path)
    else:
        saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0
    high_correct_prediction, high_total_prediction = 0, 0
    low_correct_prediction, low_total_prediction = 0, 0

    hard_total_prediction, hard_correct_prediction = 0, 0

    b_correct_predictions = [0 for _ in model.branch_predictions]
    b_taken = [0 for _ in model.branch_predictions]
    b_categories = [ [] for _ in model.branch_predictions ]
    b_images = [ {} for _ in model.branch_predictions ]

    class_correct_prediction = [ 0 for _ in range(0, model.num_classes) ]
    class_total_prediction = [ 0 for _ in range(0, model.num_classes) ]

    hard_class_correct_prediction = [ 0 for _ in range(0, model.num_classes) ]
    hard_class_total_prediction = [ 0 for _ in range(0, model.num_classes) ]

    b_class_correct_prediction = [ [0 for _ in range(0, model.num_classes) ] for _ in range(0, model.num_branches) ]
    b_class_total_prediction = [ [0 for _ in range(0, model.num_classes) ] for _ in range(0, model.num_branches) ]

    switch_class_prediction = [ [0 for _ in range(0, model.num_classes)] for _ in range(0, model.num_branches) ]

    image_idx = 0
    image_map = {}
    incorrect_images = [ [] for _ in range(0, model.num_classes) ]
    confusion_images = [ [] for _ in range(0, model.num_classes) ]
    incorrect_labels = [ [] for _ in range(0, model.num_classes) ]
    switch_confusion_images = [ [] for _ in range(0, model.num_classes) ]
    correct_images = [ [] for _ in range(0, model.num_classes) ]
    switch_low_images = [ [] for _ in range(0, model.num_classes) ]

    for batch_num in six.moves.range(eval_batch_count):
      (summaries, loss, predictions, hard_predictions,
       branch_predictions, switch_predictions,
       truth, curr_images, train_step) = sess.run(
          [model.summaries, model.cost, model.predictions,
           model.hard_predictions, model.branch_predictions,
           model.switch_predictions, model.labels,
           model.images, model.global_step])

      truth = np.argmax(truth, axis=1)

      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

      hard_predictions = np.argmax(hard_predictions, axis=1)
      hard_correct_prediction += np.sum(truth == hard_predictions)

      hard_total_prediction += hard_predictions.shape[0]

      for c in range(0, model.num_classes):
          class_correct_prediction[c] += \
                np.sum(np.multiply(truth == predictions, truth == c))
          class_total_prediction[c] += np.sum(truth == c)

      for c in range(0, model.num_classes):
          hard_class_correct_prediction[c] += \
                np.sum(np.multiply(truth == hard_predictions, truth == c))
          hard_class_total_prediction[c] += np.sum(truth == c)

      for b in range(0, batch_size):
          image_map[image_idx + b] = curr_images[b]

      for b in range(0, batch_size):
          if (hard_predictions[b] == truth[b]):
              correct_images[truth[b]].append(curr_images[b])
          else:
              confusion_images[hard_predictions[b]].append(curr_images[b])
              incorrect_images[truth[b]].append(curr_images[b])
              incorrect_labels[truth[b]].append(hard_predictions[b])

      switch_low_confidence = np.max(switch_predictions, axis=1) < \
                                     switch_confidence_thresh

      for b in range(0, batch_size):
          if(switch_low_confidence[b] == 1):
              switch_confusion_images[truth[b]].append(curr_images[b])

      switch_high_confidence = np.max(switch_predictions, axis=1) >= \
                                     switch_confidence_thresh

      low_confidence_correct = np.sum(truth[switch_low_confidence] == \
                                      hard_predictions[switch_low_confidence])

      high_confidence_correct = np.sum(truth[switch_high_confidence] == \
                                      hard_predictions[switch_high_confidence])

      high_correct_prediction += np.sum(high_confidence_correct)
      high_total_prediction += np.sum(switch_high_confidence)
      low_correct_prediction += np.sum(low_confidence_correct)
      low_total_prediction += np.sum(switch_low_confidence)

      switch_predictions = np.argmax(switch_predictions, axis=1)

      for branch_id in range(0, len(branch_predictions)):
        b_taken[branch_id] += np.sum(switch_predictions == branch_id)
        b_categories[branch_id] += (truth[switch_predictions == branch_id]).tolist()

        for c in range(0, model.num_classes):
            switch_class_prediction[branch_id][c] += \
                np.sum(np.multiply(switch_predictions == branch_id, truth == c))

      for branch_id in range(0, len(branch_predictions)):
        b_predictions_idx = np.argmax(branch_predictions[branch_id], axis=1)

        for i in range(0, len(b_predictions_idx)):
            b_images[branch_id][image_idx + i] = \
                branch_predictions[branch_id][i, b_predictions_idx[i]]

        b_correct_predictions[branch_id] += np.sum(truth == b_predictions_idx)
        for c in range(0, model.num_classes):
          b_class_correct_prediction[branch_id][c] += \
                np.sum(np.multiply(truth == b_predictions_idx, truth == c))
          b_class_total_prediction[branch_id][c] += np.sum(truth == c)

      image_idx += batch_size

    precision = 1.0 * correct_prediction / total_prediction
    hard_precision = 1.0 * hard_correct_prediction / hard_total_prediction
    best_precision = max(precision, best_precision)

    break

  b_top_200 = [ [] for _ in model.branch_predictions ]
  for branch_id in range(0, len(branch_predictions)):
      for idx in heapq.nlargest(200, b_images[branch_id], key=b_images[branch_id].get):
          b_top_200[branch_id].append(image_map[idx])

  return (b_class_correct_prediction,
          b_class_total_prediction,
          class_correct_prediction,
          class_total_prediction,
          switch_class_prediction,
          hard_class_correct_prediction,
          hard_total_prediction,
          precision,
          hard_precision,
          b_top_200,
          correct_images,
          incorrect_images,
          confusion_images,
          switch_confusion_images,
          incorrect_labels)

def evaluate(batch_size, num_classes, num_branches, num_stem_units,
             branch_base_units, uniform_cost, exploit_ratio,
             flop_ratio, weight_decay, lrn_rate, optimizer,
             dataset, eval_data_path, eval_batch_count,
             eval_dir, log_root, checkpoint_path = None,
             model_type = 'split_task'):

  """Eval loop."""
  images, labels = cifar_input.build_input(dataset, eval_data_path, batch_size, 'eval')
  model = branch_resnet_model.BranchResNet(images, labels, batch_size,
                                           num_classes, exploit_ratio, flop_ratio,
                                           num_branches, uniform_cost,
                                           num_stem_units, branch_base_units,
                                           'eval', weight_decay,
                                           lrn_rate, optimizer,
                                           model_type = model_type)
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(eval_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision = 0.0

  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    if checkpoint_path:
        saver.restore(sess, checkpoint_path)
    else:
        saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0
    hard_total_prediction, hard_correct_prediction = 0, 0
    stem_total_prediction, stem_correct_prediction = 0, 0

    b_correct_predictions = [0 for _ in model.branch_predictions]
    b_correct_switched = [0 for _ in model.branch_predictions]
    b_taken = [0 for _ in model.branch_predictions]
    b_categories = [ [] for _ in model.branch_predictions ]
    b_confidences = [ [] for _ in model.branch_predictions ]
    b_images = [ {} for _ in model.branch_predictions ]

    class_correct_prediction = [ 0 for _ in range(0, model.num_classes) ]
    class_total_prediction = [ 0 for _ in range(0, model.num_classes) ]

    hard_class_correct_prediction = [ 0 for _ in range(0, model.num_classes) ]
    hard_class_total_prediction = [ 0 for _ in range(0, model.num_classes) ]

    b_class_correct_prediction = [ [0 for _ in range(0, model.num_classes) ] for _ in range(0, model.num_branches) ]
    b_class_total_prediction = [ [0 for _ in range(0, model.num_classes) ] for _ in range(0, model.num_branches) ]

    switch_class_prediction = [ [0 for _ in range(0, model.num_classes)] for _ in range(0, model.num_branches) ]

    image_idx = 0
    image_map = {}

    for _ in six.moves.range(eval_batch_count):
      (summaries, loss, predictions, hard_predictions,
       stem_predictions,
       branch_predictions, switch_predictions,
       truth, curr_images, train_step) = sess.run(
          [model.summaries, model.cost, model.predictions,
           model.hard_predictions, model.stem_predictions,
           model.branch_predictions,
           model.switch_predictions, model.labels,
           model.images, model.global_step])

      truth = np.argmax(truth, axis=1)

      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

      hard_predictions = np.argmax(hard_predictions, axis=1)
      hard_correct_prediction += np.sum(truth == hard_predictions)
      hard_total_prediction += hard_predictions.shape[0]

      stem_predictions = np.argmax(stem_predictions, axis=1)
      stem_correct_prediction += np.sum(truth == stem_predictions)
      stem_total_prediction += stem_predictions.shape[0]

      for c in range(0, model.num_classes):
          class_correct_prediction[c] += \
                np.sum(np.multiply(truth == predictions, truth == c))
          class_total_prediction[c] += np.sum(truth == c)

      for c in range(0, model.num_classes):
          hard_class_correct_prediction[c] += \
                np.sum(np.multiply(truth == hard_predictions, truth == c))
          hard_class_total_prediction[c] += np.sum(truth == c)

      for b in range(0, batch_size):
          image_map[image_idx + b] = curr_images[b]

      switch_predictions = np.argmax(switch_predictions, axis=1)

      for branch_id in range(0, len(branch_predictions)):
        b_taken[branch_id] += np.sum(switch_predictions == branch_id)
        b_categories[branch_id] += (truth[switch_predictions == branch_id]).tolist()

        for c in range(0, model.num_classes):
            switch_class_prediction[branch_id][c] += \
                np.sum(np.multiply(switch_predictions == branch_id, truth == c))

      for branch_id in range(0, len(branch_predictions)):
        b_max_prediction = np.max(branch_predictions[branch_id], axis=1)
        for i in range(0, len(b_max_prediction)):
            b_confidences[branch_id].append(b_max_prediction[i])
        b_predictions_idx = np.argmax(branch_predictions[branch_id], axis=1)

        b_correct_switched[branch_id] += np.sum(np.multiply(switch_predictions == branch_id,
                                                 truth == b_predictions_idx))

        for i in range(0, len(b_predictions_idx)):
            b_images[branch_id][image_idx + i] = \
                branch_predictions[branch_id][i, b_predictions_idx[i]]

        b_correct_predictions[branch_id] += np.sum(truth == b_predictions_idx)
        for c in range(0, model.num_classes):
          b_class_correct_prediction[branch_id][c] += \
                np.sum(np.multiply(truth == b_predictions_idx, truth == c))
          b_class_total_prediction[branch_id][c] += np.sum(truth == c)

      image_idx += batch_size

    precision = 1.0 * correct_prediction / total_prediction
    hard_precision = 1.0 * hard_correct_prediction / hard_total_prediction
    stem_precision = 1.0 * stem_correct_prediction / stem_total_prediction
    best_precision = max(precision, best_precision)

    print(b_correct_predictions)
    for b in range(0, len(b_confidences)):
        print( sum([ c > 0.95 for c in b_confidences[b]]))
        if b_taken[b] > 0:
            print('branch_' + str(b))
            print(float(b_correct_switched[b])/b_taken[b])

    break

  b_top_200 = [ [] for _ in model.branch_predictions ]
  for branch_id in range(0, len(branch_predictions)):
      for idx in heapq.nlargest(200, b_images[branch_id], key=b_images[branch_id].get):
          b_top_200[branch_id].append(image_map[idx])

  return (b_class_correct_prediction,
          b_class_total_prediction,
          class_correct_prediction,
          class_total_prediction,
          switch_class_prediction,
          hard_class_correct_prediction,
          hard_total_prediction,
          precision,
          hard_precision,
          stem_precision,
          b_top_200)
