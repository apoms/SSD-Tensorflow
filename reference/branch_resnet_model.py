from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import layers
import cifar_input

class BranchResNet(object):
    def __init__(self, images, labels,
                 batch_size, num_classes,
                 exploit_ratio, flop_ratio,
                 num_branches, uniform_cost,
                 num_stem_units,
                 branch_base_units, mode,
                 weight_decay = 0.01,
                 learning_rate = 1e-04,
                 optimizer = 'sgd',
                 model_type = 'split_task'):

        # Network constructor
        self.batch_size = batch_size
        self.images = images
        self.labels = labels
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.exploit_ratio = exploit_ratio
        self.optimizer = optimizer
        self.mode = mode
        self.extra_train_ops = []
        self.branch_predictions = []

        # Model hyper parameters
        self.flop_ratio = flop_ratio
        self.model_type = model_type

        self.num_branches = num_branches
        self.num_stem_units = num_stem_units
        self.uniform_cost = uniform_cost
        self.branch_base_units = [branch_base_units,
                                  branch_base_units,
                                  branch_base_units]

    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.build_model()
        if self.mode == 'train':
            self.build_train_op()
        self.summaries = tf.summary.merge_all()

    def build_model(self):
        init_conv = layers.conv('init_conv', self.images, 3, 3, 16, [1, 1, 1, 1])

        stem_res = layers.residual('stem_res_0', init_conv, 16, 16, 1,
                                   self.mode == 'train', self.extra_train_ops)

        for i in range(1, self.num_stem_units):
            stem_res = layers.residual('stem_res_' + str(i), stem_res, 16, 16, 1,
                                       self.mode == 'train', self.extra_train_ops)

        self.stem = stem_res

        # Predict which branchshould get the most weight
        # TODO: Reduce switch capacity

        switch_res = layers.avg_pool('switch_pool', stem_res, 2, [1, 2, 2, 1])

        switch_sizes = [16, 32, 64]
        for i in range(0, len(switch_sizes) - 1):
            switch_res = layers.residual('switch_res_' + str(i), switch_res,
                                         switch_sizes[i], switch_sizes[i + 1], 2,
                                         self.mode == 'train',
                                         self.extra_train_ops)

        switch_bn = layers.batch_norm('switch_bn', switch_res,
                                      self.mode == 'train',
                                      self.extra_train_ops)

        switch_relu = layers.relu('switch_relu', switch_bn)

        switch_fc = layers.fully_connected('switch_fc', switch_relu,
                                           self.batch_size,
                                           self.num_branches)

        stem_bn = layers.batch_norm('stem_bn', switch_res,
                                      self.mode == 'train',
                                      self.extra_train_ops)

        stem_relu = layers.relu('stem_relu', stem_bn)

        stem_fc = layers.fully_connected('stem_fc', stem_relu,
                                          self.batch_size,
                                          self.num_classes)

        self.stem_predictions = layers.softmax('stem_predictions', stem_fc)
        log_stem_predictions = tf.nn.log_softmax(stem_fc)

        self.switch_logits = switch_fc
        self.stem_logits = stem_fc

        switch = layers.softmax('switch', switch_fc)
        log_switch = tf.nn.log_softmax(switch_fc)

        self.switch_predictions = switch

        self.exp_ratio = tf.constant(self.exploit_ratio, tf.float32)
        tf.summary.scalar('exploit_ratio', self.exp_ratio)

        split_switch = tf.split(switch, self.num_branches, axis = 1)
        log_split_switch = tf.split(log_switch, self.num_branches, axis = 1)

        masked_preds = []
        branch_sizes = [16, 16, 16]

        branch_losses = []
        scaled_branch_preds = []
        branch_usages = []

        for i in range(0, self.num_branches):
            branch_res = stem_res

            branch_num_units = self.branch_base_units
            if not self.uniform_cost:
                branch_num_units = [ (i+1) * n for n in self.branch_base_units ]

            branch_picked = tf.equal(tf.argmax(switch, axis=1),
                                  tf.cast(tf.fill([self.batch_size], i), tf.int64))

            branch_taken = tf.div(tf.reduce_sum(tf.cast(branch_picked, tf.float32)), self.batch_size)

            branch_usage = tf.div(tf.reduce_sum(split_switch[i]), self.batch_size)

            branch_loss = 0.0

            if not self.uniform_cost:
                max_cost = (self.num_branches) * self.branch_base_units[0]
                branch_loss = float((i + 1) * self.branch_base_units[0])/max_cost

            branch_usages.append(tf.reduce_sum(split_switch[i]) * branch_loss)

            branch_name = 'branch_' + str(i)

            for u in range(0, len(branch_num_units) - 1):
                branch_res = layers.residual(branch_name + '_unit_' + str(u+1) + '_0',
                                             branch_res,
                                             branch_sizes[u], branch_sizes[u+1], 2,
                                             self.mode == 'train',
                                             self.extra_train_ops)

                for u_i in range(1, branch_num_units[u]):
                    branch_res = layers.residual(branch_name + '_unit_' + str(u+1) + '_' + str(u_i),
                                                 branch_res,
                                                 branch_sizes[u+1], branch_sizes[u+1], 1,
                                                 self.mode == 'train',
                                                 self.extra_train_ops)


            final_bn = layers.batch_norm(branch_name + '_final_bn',
                                         branch_res,
                                         self.mode == 'train',
                                         self.extra_train_ops)

            branch_relu = layers.relu('branch_relu', final_bn)

            branch_pool = layers.global_avg_pool('branch_pool', branch_relu)

            fc = layers.fully_connected(branch_name + '_fc',
                                        branch_pool,
                                        self.batch_size,
                                        self.num_classes)

            tf.summary.histogram('split_pred_' + str(i), split_switch[i])
            tf.summary.scalar('branch_taken_' + str(i), branch_taken)

            #rand_vals = tf.random_uniform([self.batch_size, 1])
            #exp_factor = (1 - self.exp_ratio) * split_switch[i] + self.exp_ratio
            #switch_mask = tf.cast(tf.less(rand_vals, exp_factor), tf.float32)

            #tf.summary.scalar('switch_mask_' + str(i), switch_mask)

            logits = fc + tf.exp(log_stem_predictions)

            preds = layers.softmax(branch_name + '_softmax', logits)
            log_preds = tf.nn.log_softmax(logits)

            self.branch_predictions.append(preds)

            branch_class_mask = tf.constant([1 for c in xrange(0, self.num_classes)])
            if self.model_type == 'split_task' or self.model_type == 'split_task_semantic':
                assert(self.num_classes%self.num_branches == 0)
                num_classes_per_branch = int(self.num_classes/self.num_branches)
                branch_class_mask = tf.constant([ int(int(c/num_classes_per_branch) == i) * 0.8 + 0.1 \
                                                    for c in xrange(0, self.num_classes) ])
            if self.model_type == 'ensemble':
                scaled_preds = tf.exp(log_preds) * 1.0/self.num_branches
            else:
                scaled_preds = (tf.exp(log_preds + log_split_switch[i]) \
                               * tf.cast(branch_class_mask, tf.float32) * (1.0 - self.exp_ratio)) + \
                               (tf.exp(log_preds) * 1.0/self.num_branches * self.exp_ratio)

            branch_masked_pred = preds * tf.reshape(tf.cast(branch_picked, tf.float32),
                                                    [self.batch_size, 1])

            masked_preds.append(branch_masked_pred)
            scaled_branch_preds.append(scaled_preds)

        subclass_flat = [item for sublist in cifar_input.cifar_subclasses for item in sublist]
        logit_permute = [subclass_flat.index(c) for c in cifar_input.cifar_100_classes]
        inv_logit_permute = [cifar_input.cifar_100_classes.index(c) for c in subclass_flat]

        if self.model_type == 'split_task' or self.model_type == 'split_task_semantic':
            if self.model_type == 'split_task_semantic':
                temp_preds = tf.transpose(tf.add_n(scaled_branch_preds))
                combined_preds = tf.transpose(tf.gather(temp_preds, logit_permute))
                temp_preds_hard = tf.transpose(tf.add_n(masked_preds))
                combined_preds_hard = tf.transpose(tf.gather(temp_preds_hard, logit_permute))
            else:
                combined_preds = tf.add_n(scaled_branch_preds)
                combined_preds_hard = tf.add_n(self.branch_predictions)
            self.predictions = combined_preds
            self.hard_predictions = combined_preds_hard
        elif self.model_type == 'ensemble':
            combined_preds = tf.add_n(scaled_branch_preds)
            self.predictions = combined_preds
            self.hard_predictions = combined_preds
        elif self.model_type == 'specialist':
            combined_preds = tf.add_n(scaled_branch_preds)
            self.predictions = combined_preds
            self.hard_predictions = tf.add_n(masked_preds)

        with tf.variable_scope('costs'):
            self.cost = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(combined_preds), axis=1))

            xent = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.stem_logits, labels=self.labels)
            self.cost += tf.reduce_mean(xent)

            tf.summary.scalar('stem_loss', tf.reduce_mean(xent))

            # Flop cost
            if not self.uniform_cost:
                branch_flop_cost = tf.add_n(branch_usages)
                self.cost += branch_flop_cost * self.flop_ratio
                tf.summary.scalar('flop_loss', branch_flop_cost * self.flop_ratio)

            # L2 regularization
            self.cost += self.decay()

            tf.summary.scalar('total_loss', self.cost)

    def decay(self):
        #L2 weight regularization
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weights') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.histogram_summary(var.op.name, var)

        return tf.multiply(self.weight_decay, tf.add_n(costs))

    def build_train_op(self):
        self.lrn_rate = tf.constant(self.learning_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        if self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        elif self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lrn_rate, epsilon=1e-03)

        apply_op = optimizer.apply_gradients(
                    zip(grads, trainable_variables),
                    global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self.extra_train_ops

        self.train_op = tf.group(*train_ops)
