import numpy as np
from esFCM.utils import neighborhood_conv, rescale_between_a_b, create_window_3D, ssim3D, BiasCorrection
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import softmax
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel
import abc



def centralize_image(img, maxas=128, border=None):
    """
    Put image in the center
    :param img:
    :param maxas:
    :param border:
    :return:
    """

    n = img.shape
    if type(maxas) != list:
        maxas = [maxas, maxas, maxas]
    pads = np.array([maxas[i] - a for i, a in enumerate(n)])
    pads_r = pads // 2
    pads_l = pads - pads_r
    npads_l = pads_l * -1
    npads_r = pads_r * -1
    if border is None:
        border = img[0, 0, 0]
    new_img = np.ones((maxas[0], maxas[1], maxas[2])) * border

    pads_r[pads_r < 0] = 0
    pads_l[pads_l < 0] = 0
    npads_l[npads_l < 0] = 0
    npads_r[npads_r < 0] = 0
    # print(pads_l, pads_r)
    new_img[pads_r[0]:maxas[0] - pads_l[0], pads_r[1]:maxas[1] - pads_l[1], pads_r[2]:maxas[2] - pads_l[2]] = img[
                                                                                                              npads_r[
                                                                                                                  0]:n[
                                                                                                                         0] -
                                                                                                                     npads_l[
                                                                                                                         0],
                                                                                                              npads_r[
                                                                                                                  1]:n[
                                                                                                                         1] -
                                                                                                                     npads_l[
                                                                                                                         1],
                                                                                                              npads_r[
                                                                                                                  2]:n[
                                                                                                                         2] -
                                                                                                                     npads_l[
                                                                                                                         2]]
    return new_img, [pads, pads_l, pads_r, npads_l, npads_r, n]


class FCM(object):

    def __init__(self, parent=None):
        self.parent = parent


    @abc.abstractmethod
    def Update_memership(self):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def Update_centers(self):
        raise NotImplementedError("Subclass should implement this.")

    def predict(self, use_softmax=False):
        raise NotImplementedError("Subclass should implement this.")


    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError("Subclass should implement this.")

    def _normalizeAtlasCreateMask(self):
        raise NotImplementedError("Subclass should implement this.")

    def initialize_fcm(self, initialization_method='otsu'):

        if initialization_method=='random':
            self.rng = np.random.default_rng(0)
            m, n, o = self.image.shape
            self.Membership = self.rng.uniform(size=(m, n, o, self.num_tissues))
            self.Membership = self.Membership / np.tile(self.Membership.sum(axis=-1)[...,np.newaxis], self.num_tissues)

            self.Membership[~self.mask, :] = 0
            mask = self.mask.copy()

            numerator = np.einsum('il->l',
                                  np.expand_dims(self.image, -1)[mask, :] * pow(self.Membership[mask, :],
                                                                                         self.fuzziness))

            denominator = np.einsum('il->l', pow(self.Membership[mask], self.fuzziness))
            ind_non_denom = denominator != 0
            numerator[ind_non_denom] = numerator[ind_non_denom] / denominator[ind_non_denom]
            numerator[numerator == 0] = 0.00001
            self.Centers = numerator
        elif initialization_method=='otsu':
            from skimage.filters import threshold_multiotsu, threshold_otsu
            self.Centers = list(threshold_multiotsu(self.image[self.mask], classes=self.num_tissues+1))
            el = -2. / (self.fuzziness - 1)

            numerator = np.zeros((*self.image.shape, self.num_tissues))

            for i in range(self.num_tissues):
                numerator[self.mask, i] = np.power(abs(self.image[self.mask] - self.Centers[i])+1e-7, el)
                #numerator[self.mask, i] = np.power(abs(self.image[self.mask] - self.Centers[i])+1e-7, el)

            sumn = numerator.sum(-1)
            ind_non_zero = sumn != 0
            sumn = np.expand_dims(sumn, -1)
            numerator[ind_non_zero, :] /= sumn[ind_non_zero, :]
            self.Membership = numerator
            self.Membership_freeB = numerator.copy()

        elif initialization_method == 'kmeans':
            from sklearn.cluster import KMeans
            km = KMeans(self.num_tissues, random_state=0).fit(self.image[self.image > 0].reshape(-1, 1))
            self.Centers = km.cluster_centers_.squeeze()
            #idx = np.arange(self.num_gray)
            #c_mesh, idx_mesh = np.meshgrid(self.Centers, idx)
            el = -2. / (self.fuzziness - 1)

            numerator = np.zeros((*self.image.shape, self.num_tissues))


            for i in range(self.num_tissues):
                numerator[self.mask>0, i] = np.power(abs(self.image[self.mask>0] - self.Centers[i]),el)

            sumn = numerator.sum(-1)
            ind_non_zero = sumn != 0
            sumn = np.expand_dims(sumn, -1)
            numerator[ind_non_zero, :] /= sumn[ind_non_zero, :]
            self.Membership = numerator





class esFCM(FCM):

    def __init__(self, image, affine
                 , num_tissues,max_iter,
                 padding=0,
                  correct_bias=True,
                 mask=None, max_fail=4,fuzziness=3, epsilon = 5e-3,):
        super(esFCM, self).__init__()
        self.biascorrection = BiasCorrection()

        self.mask = mask
        self.image = image
        self.max_fail = max_fail
        self.window = create_window_3D(11, 1)

        self.estimate = image.copy()  # wstep
        self.weight = image.copy()  # wstep
        self.type_im = 'T1'

        self.num_tissues = num_tissues
        self.fuzziness = fuzziness
        self.padding = padding
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.correct_bias = correct_bias

        self.shape = image.shape  # image shape

        self.affine = affine
        self.biasfield = 1



    def SetBiasField(self, biasfield):
        self.biasfield = biasfield

    def Update_membership(self, constraint=True):
        """
        Updating FCM membership function
        @param final:
        @param membership_p:
        @return:
        """

        el = -2. / (self.fuzziness - 1)

        numerator = np.zeros_like(self.Membership)

        for i in range(self.num_tissues):
            weight = 1
            numerator[self.mask, i] = weight * np.power(abs(self.filtered_image[self.mask] - self.Centers[i]) + 1e-7,
                                                        el)

        sumn = numerator.sum(-1)
        numerator[sumn != 0] = (numerator[sumn != 0]) / ((sumn[sumn != 0])[..., np.newaxis])

        ind_non_zero_maks = self.mask == 1

        mrf_energy = self._proximity_measure(ind_non_zero_maks, self.Membership)

        numerator *= mrf_energy




        return numerator


    def Update_centers(self):
        """
        Update center of the clusters
        @return:
        """
        mask = self.mask.copy()

        numerator = np.einsum('il->l',
                              np.expand_dims(self.filtered_image, -1)[mask, :] * pow(self.Membership[mask, :],
                                                                                     self.fuzziness))
        denominator = np.einsum('il->l', pow(self.Membership[mask], self.fuzziness))
        ind_non_denom = denominator != 0
        numerator[ind_non_denom] = numerator[ind_non_denom] / denominator[ind_non_denom]

        numerator[numerator == 0] = 0.00001  # for numerical stability
        return numerator

    def WStep(self):
        """
        WStep
        @return:
        """

        input_inp = self.image

        a = self.predict()
        uq = np.unique(a)
        uq = [u for u in uq if u != 0]
        pred = np.zeros((*a.shape, self.num_tissues))
        for i, u in enumerate(uq):
            ind = a == u
            pred[ind, i] = 1

        numstd = np.einsum('ijkl,ijkl->ijk', self.Membership, (input_inp[..., None] - self.Centers) ** 2)
        denominator = np.sqrt(numstd)

        numerator = np.einsum('ijkl,l', self.Membership, self.Centers)  # +self.Membership*self.Centers

        ind_non_zero = denominator != 0

        self.weight[ind_non_zero] = denominator[ind_non_zero]
        self.estimate[ind_non_zero] = numerator[ind_non_zero]  # / denominator[ind_non_zero]

        self.weight[(~ind_non_zero)] = self.padding
        self.estimate[(~ind_non_zero)] = self.padding

    def BStep(self, mask=None):
        # bias correction step

        if mask is None:
            mask = self.mask

        mask[~self.mask] = 0
        self.biascorrection.set_info(target=self.image, reference=self.estimate,
                                     weight=self.weight, biasfield=self.biasfield, padding=self.padding,
                                     mask=mask, affine=self.affine)

        self.filtered_image = self.image.copy()
        if mask.sum() > 100:
            self.biascorrection.Run()
            self.biascorrection.Apply(self.filtered_image)
            self.filtered_image[~self.mask] = 0
        return


    def _proximity_measure(self, index_, Membership=None, sqr2dist=False):
        # Fuzzy c-means clustering with spatial information for image segmentation

        if Membership is None:
            Membership = self.Membership

        in_out = np.zeros_like(Membership)
        for i in range(self.num_tissues):
            in_out[index_, i] = \
            neighborhood_conv(Membership[..., i][..., None], kerenel_size=3, direction='xyz', sqr2dist=False)[
                index_, 0]

        in_out /= in_out.max()
        return in_out





    def fit(self):

        if not hasattr(self, 'Membership'):
            self.Membership = self.atlas_ims.copy()

        degree = 2

        biasf = PolynomialFeatures(degree)  # SplineTransformer(n_knots=2, degree=degree)#
        best_cost = -np.inf
        self.SetBiasField(biasf)
        num_fails = 0
        self.filtered_image = self.image.copy()
        old_cost = np.inf

        i = 0



        while True:
            # if i == 0:
            self.Centers = self.Update_centers()

            old_u = np.copy(self.Membership)
            self.Membership = self.Update_membership()



            cost = np.sum(abs(self.Membership - old_u) > 0.1) / np.prod(self.image[self.mask].shape)


            if cost < self.epsilon and self.correct_bias or abs(old_cost - cost) < 1e-6:

                self.WStep()

                # Apply mapping

                s1 = sobel(self.image)
                s2 = sobel(self.predict())
                fast_method = True
                if fast_method:
                    cost_ssim, ssim_map = ssim(s1 / s1.max(), s2 / s2.max(), full=True,
                                           win_size=11)
                else: # faster
                    ssim_map = ssim3D(s1 / s1.max(), s2 / s2.max(), self.window,
                                  self.window.shape[-1], 1, contrast=False)
                    cost_ssim = ssim_map[self.mask].mean()


                ssim_map = rescale_between_a_b(-ssim_map, -1000, 1000)
                ssim_map[~self.mask] = 0

                if (cost_ssim - best_cost) > 1e-4:
                    print("best SSIM value {}".format(cost_ssim))

                    self.BestCenters = self.Centers.copy()
                    self.BestFilter = self.filtered_image.copy()
                    self.BestMS = self.Membership.copy()


                    best_cost = cost_ssim
                    num_fails = 0
                else:
                    num_fails += 1

                if num_fails > self.max_fail:  # abs(old_cost_ssim - cost_ssim) < 1e-4
                    break
                if num_fails == 0:

                    self.weight = ssim_map  # rescale_between_a_b(sobel(self.image),-1,1) #ssim_map
                    self.BStep(mask=None)
                else:
                    self.filtered_image = self.BestFilter.copy()


            print("Iteration %d : cost = %f" % (i, cost))
            old_cost = cost
            if i > self.max_iter - 1:
                break

            # break
            i += 1

        ### Update with the best parameters

        self.Centers = self.BestCenters
        self.filtered_image = self.BestFilter

        self.Membership = self.BestMS


        sortedC = self.Centers.argsort()
        sorted_el = [sortedC[i] for i in range(self.num_tissues)]
        self.Membership = self.Membership[..., sorted_el]
        self.Centers = self.Centers[sorted_el]
        if self.num_tissues == 3:

            if self.type_im=='T1':  # T1
                self.wmlabel = [sortedC[2].item()]
                self.gmlabel = [sortedC[1].item()]
                self.csflabel = [sortedC[0].item()]

        elif self.num_tissues == 2:
            self.wmlabel = [sortedC[1]]
            self.gmlabel = [sortedC[0]]





    def predict(self, use_softmax=False, Membership=None):
        """
        Segment image
        @return:
        """
        if Membership is None:
            Membership = self.Membership
        if use_softmax:
            MM = softmax(Membership, -1)
        else:
            MM = Membership

        sumu = Membership.sum(-1)
        ind_zero = sumu == 0
        maxs = MM.argmax(-1)  # defuzzify
        self.output = maxs + 1
        self.output[ind_zero] = 0
        return self.output




class FCM_pure(FCM):
    """
    Fuzzy c-means clustering with spatial information for image segmentation: 2006
    """
    def __init__(self, image, affine, num_tissues, max_iter,
                 fuzziness=3, epsilon = 5e-3,padding=0,mask =None):
        super(FCM_pure).__init__()

        self.mask = mask
        self.image = rescale_between_a_b(image, 0, 1000)
        self.num_tissues = num_tissues
        self.fuzziness = fuzziness
        self.padding = padding
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.shape = image.shape  # image shape
        self.affine = affine





    def Update_memership(self):

        el = -2. / (self.fuzziness - 1)

        numerator = np.zeros_like(self.Membership)

        for i in range(self.num_tissues):
            numerator[self.mask, i] = np.power(abs(self.image[self.mask] - self.Centers[i]) + 1e-7,
                                                        el)


        sumn = numerator.sum(-1)
        numerator[sumn != 0] = (numerator[sumn != 0]) / ((sumn[sumn != 0])[..., np.newaxis])

        return numerator


    def Update_centers(self):
        """
        Update center of the clusters
        @return:
        """
        mask = self.mask.copy()

        numerator = np.einsum('il->l', np.expand_dims(self.image, -1)[mask, :] * pow(self.Membership[mask, :],
                                                                                              self.fuzziness))

        denominator = np.einsum('il->l', pow(self.Membership[mask,:], self.fuzziness))
        ind_non_denom = denominator != 0
        numerator[ind_non_denom] = numerator[ind_non_denom] / denominator[ind_non_denom]
        numerator[numerator == 0] = 0.00001
        return numerator




    def fit(self):

        oldd = np.inf
        i = 0
        while True:

            self.Centers = self.Update_centers()

            old_u = np.copy(self.Membership)

            self.Membership = self.Update_memership()

            d = np.sum(abs(self.Membership - old_u) > 0.1) / np.prod(self.image[self.mask].shape)

            print("Iteration %d : cost = %f" % (i, d))

            if d < self.epsilon  or abs(oldd - d) < 1e-2:
                break

            oldd = d

            i += 1


        self.predict()

    def predict(self, use_softmax=False):
        """
        Segment image
        @return:
        """
        Membership = self.Membership
        if use_softmax:
            MM = softmax(Membership, -1)
        else:
            MM = Membership

        sumu = Membership.sum(-1)
        ind_zero = sumu == 0
        maxs = MM.argmax(-1)  # defuzzify
        self.output = maxs + 1
        self.output[ind_zero] = 0
        return self.output




