"""
Détection d'oiseaux et analyse des détections

Chronologie :
- 2024-04-15 : première version pour essayer BirdNet
- 2024-04-27 : ajout d'une boucle pour détecter les oiseaux dans plusieurs fichiers audio
- 2024-04-28 : ajout d'une fonction pour générer des figures

Crédit :
- Noé Aubin-Cadot

To do :
- Faire une loop pour analyser tout les fichiers WAV Merlin depuis le début et générer des statistiques de détections depuis le début de toutes les observations.

Dependencies :
- pip install birdnetlib
- pip install librosa
- pip install resampy
- pip install adjustText

--------------------------------------------------------

BirdNet Analyser :

Source :
	https://github.com/kahst/BirdNET-Analyzer?tab=readme-ov-file

Installation :
	pip install birdnetlib
	pip install librosa
	pip install resampy

Directory de birdnetlib :
	/usr/local/lib/python3.9/site-packages/birdnetlib

Questions :
- Est-ce que je peux les générer en français ?
Réponse :
- Non ça ne génère que les noms en anglais, il faut traduire les oiseaux soi-même.

--------------------------------------------------------

"""


################################################################################
################################################################################
# Importer des librairies

import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta

################################################################################
################################################################################
# Définition de fonctions
# 2024-04-15

def extract_birds(
	input_file,
	output_file_detections,
	output_file_uniques,
	verbose = False,
):
	# https://github.com/kahst/BirdNET-Analyzer?tab=readme-ov-file
	if verbose:
		print('Importing libraries...')
	from birdnetlib import Recording
	from birdnetlib.analyzer import Analyzer
	# Load and initialize the BirdNET-Analyzer models.
	analyzer = Analyzer()
	recording = Recording(
		analyzer = analyzer,
		path     = input_file, # Fichier venant de Merlin via AirDrop
		lat      =  45.50884,
		lon      = -73.58781,
		date     =  datetime(year=2023, month=5, day=13),
		min_conf =  0.05,
	)
	recording.analyze()
	l_detections = recording.detections
	df_detections = pd.DataFrame(l_detections)
	df_detections.to_csv(output_file_detections,index=False)
	df_detections.sort_values(by='confidence',ascending=False,inplace=True)
	df_detections.drop_duplicates(subset='common_name',inplace=True)
	print(df_detections)
	if verbose:
		print('Exporting :',output_file)
	df_detections.to_csv(output_file_uniques,index=False)

	"""
	À implémenter :
	- un dataset de toutes les observations avec start_time et end_time en datetime et pas que en secondes
	- un dataset des oiseaux distincts avec le nombre d'occurences et la moyenne du confidence
	- traduction vers le français
	"""

################################################################################
################################################################################
# Définition de fonctions
# 2024-04-27

# Définition d'une fonction qui prend un fichier d'entrée et rend la date correspondante
def input_file_to_date(
	input_file,
):
	date_str = input_file.split('/')[-1].split('.')[0].split(' ')[0]
	return date_str

# Définition d'une fonction qui prend une liste de fichiers audio et rend un DataFrame des oiseaux détectés
def input_files_to_DataFrame(
	input_files,
	latitude  = 45.50884,  # Latitude de Montréal
	longitude = -73.58781, # Longitude de Montréal
	min_conf  = 0.05,      # Niveau de confiance minimal
	date      = None,      # Si on ne veut analyser qu'une date spécifique (format string 'YYYY-MM-DD')
):
	# Librairies pour BirdNet
	from birdnetlib import Recording
	from birdnetlib.analyzer import Analyzer

	# Si une date est spécifiée, on filtre les fichiers à analyser
	if date:
		input_files_to_analyse = [input_file for input_file in input_files if input_file_to_date(input_file)==date]

	# On déclare une liste dans laquelle on va mettre les DataFrames des détections d'oiseaux
	list_of_df = []

	# Un compteur pour spécifier où la boucle est rendue
	counter = 0

	# On fait une boucle sur les fichiers audio
	for input_file in sorted(input_files_to_analyse):
		
		# On affiche où on est rendu
		counter += 1
		print(f'\nFichier audio {counter}/{len(input_files_to_analyse)}.')
		print('input_file =',input_file)

		# On crée un objet datetime
		datetime_str = input_file.split('/')[-1].split('.')[0]
		datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H%M")

		# Détection des oiseaux via BirdNet
		recording = Recording(
			analyzer = Analyzer(),
			path     = input_file,
			lat      = latitude,
			lon      = longitude,
			date     = datetime_obj,
			min_conf = min_conf,
		)
		recording.analyze()

		# On prend la liste des nouvelles détections
		l_detections = recording.detections

		# On affiche le nombre de détections pour ce fichier audio
		print('Nombre de détections :',len(l_detections))

		# Conversion de la liste de détections en DataFrame
		df_detections = pd.DataFrame(l_detections)

		# On ajoute une colonne qui spécifie la durée de la détection
		df_detections['duration'] = df_detections['end_time'] - df_detections['start_time']

		# On ajoute une colonne qui spécifie la date et le datetime du fichier
		df_detections['file_date']     = datetime_obj.strftime('%Y-%m-%d')
		df_detections['file_datetime'] = datetime_obj.strftime('%Y-%m-%d %H:%M')

		# On ajoute une colonne qui spécifie le datetime de début et de fin de détection de l'oiseau
		df_detections['bird_datetime_start'] = df_detections['start_time'].apply(lambda x:(datetime_obj+timedelta(seconds=x)).strftime('%Y-%m-%d %H:%M'))
		df_detections['bird_datetime_end']   = df_detections['end_time'].apply(lambda x:(datetime_obj+timedelta(seconds=x)).strftime('%Y-%m-%d %H:%M'))       

		# On ajoute les oiseaux détectés à la liste de détections d'oiseaux
		list_of_df.append(df_detections)

	# Génération d'un DataFrame qui contient toutes les détections
	df_detections = pd.concat(
		objs = list_of_df,
		axis = 0,
		ignore_index = True,
	)

	# On réordonne les colonnes
	cols = [
		'file_datetime',
		'file_date',
		'bird_datetime_start',
		'bird_datetime_end',
		'start_time',
		'end_time',
		'duration',
		'confidence',
		'common_name',
		'scientific_name',
		'label',
	]
	df_detections = df_detections[cols]

	# On rend le DataFrame
	return df_detections


def translate_birds(
	input_file_detections,
	input_file_traductions,
	output_file,
):
	# On importe les données des détections
	print('Importing :',input_file_detections)
	df_detections  = pd.read_csv(input_file_detections)

	# On importe les données des traductions
	print('Importing :',input_file_traductions)
	df_traductions = pd.read_csv(input_file_traductions)

	# On regarde les oiseaux détectés qui ne sont pas dans les oiseaux dont la traduction est connue
	set_detected   = set(df_detections['common_name'].sort_values().drop_duplicates())
	set_translated = set(df_traductions['oiseau_en'].sort_values())
	birds_not_translated = sorted(set_detected-set_translated)
	if len(birds_not_translated)>0:
		print('Oiseaux non traduits :')
		for bird in birds_not_translated:
			print(bird)
		raise Exception('Translations missing... please fill in the translation file for the birds mentioned above.')

	# On traduit par une jointure
	df_traductions.rename(columns={'oiseau_en':'common_name'},inplace=True)

	# On fait une jointure
	df_detections = pd.merge(
		left  = df_detections,
		right = df_traductions,
		on    = 'common_name',
		how   = 'left',
	)

	# On exporte
	print('Exporting :',output_file)
	df_detections.to_csv(output_file,index=False)
	print('Done.')

def analyse_birds(
	input_file,
):
	# On importe les données
	print('Importing :',input_file)
	df = pd.read_csv(input_file)

	# On filtre sur la plage horaire des observations
	datetime_start,datetime_end = '2024-04-27 11:37','2024-04-27 14:15' # Domaine Saint-Paul à l'Île des Soeurs
	#datetime_start,datetime_end = '2024-04-27 14h20',None # Parc Maynard-Ferguson à l'Île des Soeurs
	if datetime_start:
		df = df[df['file_datetime']>=datetime_start]
	if datetime_end:
		df = df[df['file_datetime']<=datetime_end]

	# On ne garde que les détections qui ont au moins 0.25 de confiance
	df = df[df['confidence']>=0.25]

	import matplotlib.pyplot as plt

	do_show_confident_birds=0
	if do_show_confident_birds:
		df.sort_values(by=['oiseau_fr','confidence'],ascending=[True,False],inplace=True)
		df.drop_duplicates(subset=['oiseau_fr'],inplace=True)
		df.sort_values(by='confidence',ascending=False,inplace=True)
		df = df[['oiseau_fr','confidence']]
		print(df.to_string(index=False))
		ax = df.head(50).sort_values(by='confidence',ascending=True).plot(
			kind    = 'barh',
			x       = 'oiseau_fr',
			y       = 'confidence',
			figsize = (12,8),
		)
		ax.tick_params(axis='y', labelsize=11)
		plt.title('Confiance maximale détectée par espèce (confiance ≥ 0.25)',size=18)
		plt.xlabel('Confiance maximale détectée',size=15)
		plt.ylabel("Espèce d'oiseau",size=15)
		xticks = np.arange(0.0,1.05,0.05).round(2)
		plt.xticks(xticks,[format(x,'0.2f') for x in xticks])
		plt.xlim(0,1)
		plt.grid()
		plt.tight_layout()
		plt.show()

	do_show_frequent_birds=1
	if do_show_frequent_birds:
		s_count = df['oiseau_fr'].value_counts().sort_values(ascending=True)
		print(s_count)
		ax = s_count.plot(
			kind    = 'barh',
			figsize = (12,8),
		)
		ax.tick_params(axis='y', labelsize=11)
		plt.title('Nombre de détections (confiance ≥ 0.25)',size=18)
		plt.xlabel('Nombre de détections',size=15)
		plt.ylabel("Espèce d'oiseau",size=15)
		plt.xticks(np.arange(0,95,5),np.arange(0,95,5))
		plt.grid()
		plt.tight_layout()
		plt.show()

	do_scatterplot_agg=0
	if do_scatterplot_agg:
		# Pour chaque oiseau on calcule le nombre de fois qu'il a été détecté
		s_count = df['oiseau_fr'].value_counts().sort_index()
		s_count.name = 'count'
		# Pour calculer la confiance moyenne sur les 5 meilleures détections par oiseau
		df_sorted  = df.sort_values(by=['oiseau_fr', 'confidence'], ascending=[True, False])
		df_top     = df_sorted[['oiseau_fr','confidence']].groupby('oiseau_fr').head(5)
		s_top      = df_top.groupby('oiseau_fr')['confidence'].mean().sort_index()
		s_top.name = 'confidence_mean'
		# Concaténation du dénombrement et des confiances moyennes des 5 plus confiantes observations
		df_count_top = pd.concat((s_count,s_top),axis=1)
		print(df_count_top)
		#df_count_top = df_count_top[df_count_top['count']>=3]
		df_count_top.plot(
			kind    = 'scatter',
			x       = 'count',
			y       = 'confidence_mean',
			figsize = (12,8),
			s       = 5,
		)
		plt.title('Confiance moyenne vs nombre de détections (confiance ≥ 0.25)',size=18)
		plt.xlabel('Nombre de détections',size=15)
		plt.ylabel('Moyenne de la confiance des 5 plus confiantes détections',size=15)
		plt.xticks(np.arange(0,100,5),np.arange(0,100,5))
		yticks = np.arange(0.2,1.05,0.05).round(2)
		plt.yticks(yticks,[format(y,'0.2f') for y in yticks])
		# https://adjusttext.readthedocs.io/en/latest/_modules/adjustText.html
		bbox = {
			'facecolor' : 'red',
			'alpha'     : 0.05,
		}
		texts = [plt.text(x=x,y=y,s=bird, bbox=bbox) for x, y, bird in zip(df_count_top['count'], df_count_top['confidence_mean'], df_count_top.index.to_list())]
		from adjustText import adjust_text # pip install adjustText
		adjust_text(
			texts           = texts,
			arrowprops      = dict(arrowstyle="->", color='r', lw=0.5, mutation_scale=10),
			connectionstyle = 'arc3,rad=0.3',
		)
		plt.grid()
		plt.tight_layout()
		plt.show()


################################################################################
################################################################################
# Utilisation de fonctions

def main():

	# ------------------------------
	# Ancien flux

	# 2024-04-15 (ancien flux)
	do_extract_birds=0
	if do_extract_birds:
		#file_datetime = "2023-05-13 1303"
		file_datetime = "2024-04-15 1643"
		file_datetime = "2024-04-19 1557"
		extract_birds(
			input_file             = f"input/{file_datetime}.wav",
			output_file_detections = f"output/{file_datetime}_detections.csv",
			output_file_uniques    = f"output/{file_datetime}_uniques.csv",
		)

	# ------------------------------
	# Nouveau flux

	# 2024-04-27
	do_input_files_to_DataFrame=0
	if do_input_files_to_DataFrame:
		# On prend la liste des fichiers audio WAV disponibles
		input_files = sorted(glob.glob('input/audio/*.wav'))
		# On calcule le DataFrame des oiseaux détectés
		df_detections = input_files_to_DataFrame(
			input_files = input_files,
			date        = '2024-04-27',
		)
		# Preview du DataFrame des détections
		print('\ndf_detections.shape =',df_detections.shape)
		print(df_detections)
		df_detections.to_csv('output/2024-04-27_df_detections_local_en.csv',index=False)

	# 2024-04-27
	do_translate_birds=0
	if do_translate_birds:
		input_file_detections  = 'output/2024-04-27_df_detections_local_en.csv'
		input_file_traductions = 'traduction/translation_birds_english_to_french.csv'
		output_file            = 'output/2024-04-27_df_detections_local_fr.csv'
		translate_birds(
			input_file_detections  = input_file_detections,
			input_file_traductions = input_file_traductions,
			output_file            = output_file,
		)

	# 2024-04-27
	do_analyse_birds=0
	if do_analyse_birds:
		input_file = 'output/2024-04-27_df_detections_local_fr.csv'
		analyse_birds(
			input_file = input_file,
		)

if __name__ == '__main__':
	main()









