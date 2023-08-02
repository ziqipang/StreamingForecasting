import numpy as np


def pred_metrics(preds, gt_preds):
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err_min = err[row_idcs, min_idcs]
    ade = err_min.mean()
    fde = err_min[:, -1].mean()
    mr = np.sum(err_min[:, -1] > 2) / len(min_idcs)

    err1 = err[row_idcs, 0]
    ade1 = err1.mean()
    fde1 = err1[:, -1].mean()
    mr1 = np.sum(err1[:, -1] > 2) / len(min_idcs)
    return ade1, fde1, mr1, ade, fde, mr, min_idcs


class Formatter:
    def __init__(self) -> None:
        self.buffer = {
            'ade1': [], 'fde1': [], 'mr1': [],
            'ade': [], 'fde': [], 'mr': [],
            'loss': [], 'reg_loss': [], 'conf_loss': []
        }
        return
    
    def append(self, out, data, loss):
        # loss keys
        self.buffer['loss'] += [loss['loss'].detach().cpu().data]
        self.buffer['reg_loss'] += [loss['pred'].detach().cpu().data]
        self.buffer['conf_loss'] += [loss['conf'].detach().cpu().data]

        # metric values
        preds = [x[0].detach().cpu().numpy()[np.newaxis, :] for x in out['prediction']]
        preds = np.concatenate(preds, axis=0)
        gts = [g[0].detach().cpu().numpy()[np.newaxis, :] for g in data['gt_futures']]
        gts = np.concatenate(gts, axis=0)
        ade1, fde1, mr1, ade, fde, mr, _ = pred_metrics(preds, gts)

        self.buffer['ade'] += [ade]
        self.buffer['fde'] += [fde]
        self.buffer['mr'] += [mr]
        self.buffer['ade1'] += [ade1]
        self.buffer['fde1'] += [fde1]
        self.buffer['mr1'] += [mr1]
        return
    
    def display(self, iter_num, writer=None):
        message = 'loss %2.4f reg %2.4f conf %2.4f, ade %2.4f, fde %2.4f, mr %2.4f, ade1 %2.4f, fde1 %2.4f, mr1 %2.4f,'\
            % (np.mean(self.buffer['loss']), np.mean(self.buffer['reg_loss']), np.mean(self.buffer['conf_loss']),
               np.mean(self.buffer['ade']), np.mean(self.buffer['fde']), np.mean(self.buffer['mr']),
               np.mean(self.buffer['ade1']), np.mean(self.buffer['fde1']), np.mean(self.buffer['mr1']))
        infos = {
            'loss': np.mean(self.buffer['loss']), 'reg': np.mean(self.buffer['reg_loss']), 'conf': np.mean(self.buffer['conf_loss']),
            'ade': np.mean(self.buffer['ade']), 'fde': np.mean(self.buffer['fde']), 'mr': np.mean(self.buffer['mr']),
            'ade1': np.mean(self.buffer['ade1']), 'fde1': np.mean(self.buffer['fde1']), 'mr1': np.mean(self.buffer['mr1'])
        }
            
        # tensorboard
        if writer is not None:
            writer.add_scalar('loss/loss', np.mean(self.buffer['loss']), iter_num)
            writer.add_scalar('loss/pred', np.mean(self.buffer['reg_loss']), iter_num)
            writer.add_scalar('loss/conf', np.mean(self.buffer['conf_loss']), iter_num)
            writer.add_scalar('metric/ade', np.mean(self.buffer['ade']), iter_num)
            writer.add_scalar('metric/fde', np.mean(self.buffer['fde']), iter_num)
            writer.add_scalar('metric/mr', np.mean(self.buffer['mr']), iter_num)
        
        # renew
        self.buffer = {
            'ade1': [], 'fde1': [], 'mr1': [],
            'ade': [], 'fde': [], 'mr': [],
            'loss': [], 'reg_loss': [], 'conf_loss': []
        }
        return message, infos