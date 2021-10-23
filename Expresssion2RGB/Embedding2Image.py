import numpy as np
import cv2

class Embedding2Image:
    def scale_to_RGB(self, channel,truncated_percent):
        truncated_down = np.percentile(channel, truncated_percent)
        truncated_up = np.percentile(channel, 100 - truncated_percent)
        channel_new = ((channel - truncated_down) / (truncated_up - truncated_down)) * 255
        channel_new[channel_new < 0] = 0
        channel_new[channel_new > 255] = 255
        return np.uint8(channel_new)

    def save_transformed_RGB_to_image_and_csv(self, spot_row_in_fullres,
                                          spot_col_in_fullres,
                                          max_row, max_col,
                                          X_transformed,
                                          plot_spot_radius,
                                          ):
  
        img = np.ones(shape=(max_row + 1, max_col + 1, 3), dtype=np.uint8) * 255
        for index in range(len(X_transformed)):
            cv2.rectangle(img, (spot_col_in_fullres[index] - plot_spot_radius, spot_row_in_fullres[index] - plot_spot_radius),
                          (spot_col_in_fullres[index] + plot_spot_radius, spot_row_in_fullres[index]+ plot_spot_radius),
                          color=(int(X_transformed[index][0]), int(X_transformed[index][1]), int(X_transformed[index][2])),
                          thickness=-1)
        #optional  both/high/low/none
        expression_img = cv2.resize(img, dsize=(2000, 2000), interpolation=cv2.INTER_CUBIC)
        return expression_img
    
    def emb2img(self, adata):
        X_transform = adata.obsm["embedding"]
        full_data = adata.obs
        X_transform[:, 0] = self.scale_to_RGB(X_transform[:, 0], 100)
        X_transform[:, 1] = self.scale_to_RGB(X_transform[:, 1], 100)
        X_transform[:, 2] = self.scale_to_RGB(X_transform[:, 2], 100)
        
        radius = int(0.5 *  adata.uns['fiducial_diameter_fullres'] + 1)
        max_row = max_col = int((2000 / adata.uns['tissue_hires_scalef']) + 1)
        return self.save_transformed_RGB_to_image_and_csv(full_data['pxl_col_in_fullres'].values,
                                          full_data['pxl_row_in_fullres'].values,
                                          max_row,
                                          max_col,
                                          X_transform,
                                          plot_spot_radius = radius
                                          )

