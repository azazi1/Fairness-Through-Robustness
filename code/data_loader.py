import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import helper as hp

def plot_image(image, root_folder='adversarial_examples', subfolder='unnamed', filename='unnamed', 
               plot_title='', save_fig=True, show_fig=True, extension='png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_title(plot_title)
    ax.imshow(image, interpolation='bilinear')
    
    if save_fig:
        hp.create_dir('{}'.format(root_folder))
        hp.create_dir('{}/{}'.format(root_folder, subfolder))
        plt.savefig('{}/{}/{}.{}'.format(root_folder, subfolder, filename, extension))
    if show_fig:
        plt.show()
    plt.clf()
    plt.close()


class Dataloader:
    """
    Base class for loading data; any other dataset loading class should inherit from this to ensure consistency
    """
    TRAIN = 'train'
    TEST = 'test'
    
    def __init__(self):
        pass
    
    def length(self, portion):
        raise NotImplementedError("Implement length in child class!")

    def num_classes(self):
        raise NotImplementedError("Implement num_classes in child class!")
        
    def get_image(self, portion, idx):
        raise NotImplementedError("Implement get_image in child class!")


class UTKFace(Dataloader):
    
    """
    Class that does the dirty loading for the UTKFace dataset
    """
    
    def __init__(self, name, root_dir='.', load_filenames=True):
        #TODO: allow for specifying protected classes and target class (via optional parameters)
        #TODO: allow for different distributions of protected classes (via optional parameters)
        random.seed(42)

        directory = '{}/../data/UTKFace/'.format(root_dir) # directory with the images
        
        self.name = name
        self.data_transform = transforms.Compose([ # place any needed transforms here
                transforms.ToTensor(),
                transforms.Normalize([0., 0., 0.], [1., 1., 1.])
            ])
        
        # properties of the images
        self.ages = np.arange(1,117) # age in years
        self.classes = ['0-15', '15-25', '25-40', '40-60', '60+']  # bins of ages (see resolve_class_label())
        self.genders = np.arange(2) # (male, female)
        self.gender_id_to_label = {0: 'male', 1: 'female'} # mapping from gender id (0/1) to its string label
        self.gender_label_to_id = {v:k for k,v in self.gender_id_to_label.items()}
        self.races = np.arange(5)  # (White, Black, Asian, Indian, Others)
        self.race_id_to_label = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'other'}
        self.race_label_to_id = {v:k for k,v in self.race_id_to_label.items()}
#         self.race_id_to_label = {0: 'other', 1: 'black'}
        
        # Extract info from each image
        self.image_paths, self.image_ages, self.image_genders, self.image_races = [], [], [], []
        self.image_classes = []
        
        if load_filenames:
            self.load_paths(directory)
    
    def visual_samples(self):
        print ("First 5 train samples")
        for i in range(5):
            train_image, train_image_class, train_image_gender, train_image_race =                 self.get_image('train', int(i)), self.classes[self.get_image_label('train', int(i))],                 self.gender_id_to_label[self.get_image_protected_class('train', int(i), attr='gender')],                self.race_id_to_label[self.get_image_protected_class('train', int(i), attr='race')]
            train_image = np.moveaxis(train_image.numpy(), 0, -1)
            train_image = (train_image * self.data_transform.transforms[-1].std) +                 self.data_transform.transforms[-1].mean
            plot_image(train_image, save_fig=False,
                       plot_title='Class: {}, Gender: {}, Race: {}'.format(train_image_class,
                                                                                       train_image_gender,
                                                                                       train_image_race))
        print ("First 5 test samples")
        for i in range(5):
            test_image, test_image_class, test_image_gender, test_image_race =                 self.get_image('test', int(i)), self.classes[self.get_image_label('test', int(i))],                 self.gender_id_to_label[self.get_image_protected_class('test', int(i), attr='gender')],                self.race_id_to_label[self.get_image_protected_class('test', int(i), attr='race')]
            test_image = np.moveaxis(test_image.numpy(), 0, -1)
            test_image = (test_image * self.data_transform.transforms[-1].std) +                 self.data_transform.transforms[-1].mean
            plot_image(test_image, save_fig=False,
                       plot_title='Class: {}, Gender: {}, Race: {}'.format(test_image_class,
                                                                                       test_image_gender,
                                                                                       test_image_race))
    
    def load_paths(self, directory):
        filepath = os.fsencode(directory)
        # sort the o/p of listdir to ensure that the ordering is same always
        for file in sorted(os.listdir(filepath)):
            filename = os.fsdecode(file)
            try:
                self.image_paths = np.append(self.image_paths, directory+filename)
                age, gender, race, _ = filename.split('_')
                age, gender, race = int(age), int(gender), int(race)
                age_bin = self.resolve_class_label(age)
            except:
                pass
#                print('Error: Age, Gender, and/or Race Unknown')
               
#                img = imageio.imread(directory+filename)
#                plt.imshow(img)
#                plt.show()
               
#                print(filename)
               
#                manual_classification = input('Would you like to classify this image manually? (y/n)\n')
#                if manual_classification == 'y':
#                    age = input('Age (years old): \n')
#                    gender = input('Gender {0: male, 1: female}: \n')
#                    race = input('Race {0: white, 1: black, 2: asian, 3: indian, 4: other}: \n')
            
            self.image_classes = np.append(self.image_classes, int(age_bin))
            self.image_ages = np.append(self.image_ages, age)
            self.image_genders = np.append(self.image_genders, int(gender))
#             self.image_races = np.append(self.image_races, int(race == 1))
            self.image_races = np.append(self.image_races, int(race))
            
        # Split the data into train and test sets
        all_indices = list(range(len(self.image_paths)))
        random.shuffle(all_indices)
        
        train_cutoff = int(0.8 * len(all_indices)) # 80:20 train test split
        self.train_indices = all_indices[:train_cutoff]
        self.train_image_paths = self.image_paths[self.train_indices]
        self.train_image_classes = self.image_classes[self.train_indices].astype('int')
        self.train_image_ages = self.image_ages[self.train_indices].astype('int')
        self.train_image_genders = self.image_genders[self.train_indices].astype('int')
        self.train_image_races = self.image_races[self.train_indices].astype('int')
        
        self.test_indices = all_indices[train_cutoff:]
        self.test_image_paths = self.image_paths[self.test_indices]
        self.test_image_classes = self.image_classes[self.test_indices].astype('int')
        self.test_image_ages = self.image_ages[self.test_indices].astype('int')
        self.test_image_genders = self.image_genders[self.test_indices].astype('int')
        self.test_image_races = self.image_races[self.test_indices].astype('int')
        
        assert len(self.train_indices) + len(self.test_indices) == len(all_indices)

    def length(self, portion):
        if portion == 'train':
            return len(self.train_image_paths)
        elif portion == 'test':
            return len(self.test_image_paths)
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def num_classes(self):
        return len(self.classes)
    
    def load_image(self, image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.data_transform(image)
        return image
    
    def get_image(self, portion, idx):
        if portion == 'train':
            return self.load_image(self.train_image_paths[idx])
#             return self.loaded_images_train[idx]
        elif portion == 'test':
            return self.load_image(self.test_image_paths[idx])
#             return self.loaded_images_test[idx]
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def get_image_label(self, portion, idx):
        if portion == 'train':
            return self.train_image_classes[idx]
        elif portion == 'test':
            return self.test_image_classes[idx]
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def get_image_protected_id_to_label(self, protected_id, attr='gender'):
        if attr not in ['gender', 'race']:
            raise ValueError('{} not an acceptable attr. Must be one of {}'.format(attr, ['gender', 'race']))
        
        return self.race_id_to_label[protected_id] if attr == 'race' else self.gender_id_to_label[protected_id]
    
    def get_image_protected_label_to_id(self, protected_label, attr='gender'):
        if attr not in ['gender', 'race']:
            raise ValueError('{} not an acceptable attr. Must be one of {}'.format(attr, ['gender', 'race']))
        
        return self.race_label_to_id[protected_label] if attr == 'race' else self.gender_label_to_id[protected_label]
    
    def get_image_protected_class(self, portion, idx, attr='gender'):
        ## gender is binary, however for race we consider black as the minority and every other race as majority
        if attr not in ['gender', 'race']:
            raise ValueError('{} not an acceptable attr. Must be one of {}'.format(attr, ['gender', 'race']))

        if portion == 'train':
            return self.train_image_genders[idx] if attr == 'gender' else self.train_image_races[idx]
        elif portion == 'test':
            return self.test_image_genders[idx] if attr == 'gender' else self.test_image_races[idx]
        else:
            raise ValueError("Portion {} not understood".format(portion))
    
    def resolve_class_label(self, age):
        if age in range(15):
            age_id = 0
        elif age in range(15,25):
            age_id = 1
        elif age in range(25,40):
            age_id = 2
        elif age in range(40,60):
            age_id = 3
        elif age >= 60:
            age_id = 4
        else:
            raise ValueError("Not sure how to handle this age: {}".format(age))
        
        return age_id


def local_testing_UTKFace():
    obj = UTKFace(name='utkface')
    obj.visual_samples()

# local_testing_UTKFace()


