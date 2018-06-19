import warnings

from init_param import novelView
from prepare_data import *

warnings.filterwarnings("ignore")
from train import load_networks, read_illum_images, evaluate_system, compute_psnr
from skimage.color import rgb2hsv, hsv2rgb
import pytorch_ssim
from torch.autograd import Variable
from cv2 import imwrite


def adjust_tone(input):
    input[input > 1] = 1
    input[input < 0] = 0
    output = input ** (1 / 1.5)
    output = rgb2hsv(output)
    output[:, :, 1] = output[:, :, 1] * 1.5
    output = hsv2rgb(output)
    return output


def get_img_ind(inPos):
    ind = int(inPos * (param.origAngRes - 1))
    return ind


def write_error(estimated, reference, resultPath):
    curPSNR = compute_psnr(estimated, reference)
    estimated = Variable(torch.from_numpy(np.expand_dims(estimated,axis=3)).permute(3,2,0,1).float())
    reference = Variable(torch.from_numpy(np.expand_dims(reference,axis=3)).permute(3,2,0,1).float())
    curSSIM = pytorch_ssim.ssim(estimated, reference).data[0]

    fid = open(resultPath + '/ObjectiveQuality.txt', 'w')
    fid.write('PSNR: %3.2f\n' % curPSNR)
    fid.write('SSIM: %1.3f\n' % curSSIM)
    fid.close()


def synthesize_novel_views(depth_net, color_net, inputLF, fullLF, resultPath):
    numNovelViews = len(novelView.Y)

    # for vi in range(numNovelViews):
    for vi in range(1):
        indY = get_img_ind(novelView.Y[vi])
        indX = get_img_ind(novelView.X[vi])

        curRefPos = np.array([novelView.Y[vi], novelView.X[vi]])
        curRefPos = np.expand_dims(curRefPos, axis=1)
        curRefPos = torch.from_numpy(curRefPos).float()
        # performs the whole process of extracting features, evaluating the
        # two sequential networks and generating the output synthesized image
        print('View %02d of %02d' % (vi, numNovelViews))
        print('**********************************')
        synthesizedView = evaluate_system(depth_net, color_net, images=inputLF, refPos=curRefPos)

        synthesizedView = synthesizedView[:, :, :, -1]
        # crop the result and reference images
        curEst = crop_img(synthesizedView, 10)
        curRef = crop_img(fullLF[:, :, :, indY, indX], param.depthBorder + param.colorBorder + 10)


        # write the numerical evaluation and the final image
        if indY == 0 and indX == 0:
            write_error(curEst, curRef, resultPath)
        imwrite(resultPath + '/Images/' + ('%02d_%02d.png' % (indY, indX)), (adjust_tone(curEst) * 255).astype(int))


def test():
    # Initialization
    sceneFolder = './Scenes'
    resultFolder = './Results'

    # load the pre-trained networks
    [depth_net, color_net, _, _] = load_networks()

    # Generate novel views for each scene
    [sceneNames, scenePaths, numScenes] = get_folder_content(sceneFolder)

    for ns in range(numScenes):
        print('**********************************')
        print('Working on the ' + sceneNames[ns][0:- 4] + 'dataset')

        resultPath = resultFolder + '/' + sceneNames[ns][0:- 4]
        make_dir(resultPath + '/Images')

        print('Loading input light field ...',end = '')
        [curFullLF, curInputLF] = read_illum_images(scenePaths[ns])
        print('Done')
        print('**********************************')

        print('Synthesizing novel views')
        print('--------------------------')
        synthesize_novel_views(depth_net, color_net, curInputLF, curFullLF, resultPath)
        print()


if __name__ == "__main__":
    test()
