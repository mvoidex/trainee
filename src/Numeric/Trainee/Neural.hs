{-# LANGUAGE GADTs, FlexibleContexts, RankNTypes #-}

module Numeric.Trainee.Neural (
	LayerBuild(..),
	Layer,
	NetBuild(..), Net,
	net, input, (⭃),
	fc,

	sigma, relu,
	summator, biaser, activator,
	norm,

	module Numeric.Trainee.Types,
	module Numeric.Trainee.Learnee
	) where

import Prelude hiding ((.), id)
import Prelude.Unicode

import Control.Category
import Control.Monad (replicateM, liftM)
import Data.Random
import Numeric.LinearAlgebra

import Numeric.Trainee.Types
import Numeric.Trainee.Gradee (matVec, swap, odot, unary, Unary)
import Numeric.Trainee.Learnee

data LayerBuild m a = LayerBuild {
	layerOut ∷ Int,
	layerBuild ∷ Int → m (Layer a) }

type Layer a = Learnee (Vector a) (Vector a)

data NetBuild a = NetBuild {
	buildOut ∷ Int,
	buildNet ∷ Learnee (Vector a) (Vector a) }

type Net a = Learnee (Vector a) (Vector a)

net ∷ RVar (NetBuild a) → IO (Net a)
net act = liftM buildNet $ runRVar act StdRandom

input ∷ Monad m ⇒ Int → m (NetBuild a)
input i = return $ NetBuild i (computee id)

(⭃) ∷ Monad m ⇒ m (NetBuild a) → LayerBuild m a → m (NetBuild a)
n ⭃ l = do
	n' ← n
	l' ← layerBuild l (buildOut n')
	return $ n' {
		buildOut = layerOut l,
		buildNet = buildNet n' ⥤ l' }

-- | Fully connected layer
fc ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Unary (Vector a) → Int → LayerBuild m a
fc f outputs = LayerBuild outputs $ \inputs → do
	s ← summator outputs inputs
	b ← biaser outputs
	return $ s ⥤ b ⥤ activator f

sigma ∷ Floating a ⇒ a → a
sigma t = 1 / (1 + exp (negate t))

relu ∷ (Ord a, Num a) ⇒ a → a
relu = max 0

summator ∷ (MonadRandom m, Distribution Normal a, Fractional a, Numeric a, Num (Vector a), Parametric a) ⇒ Int → Int → m (Layer a)
summator outputs inputs = do
	ws ← runRVar (replicateM (inputs * outputs) norm) StdRandom
	return $ learnee (matVec . swap) ((outputs >< inputs) ws)

biaser ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Int → m (Learnee (Vector a) (Vector a))
biaser sz = do
	bs ← runRVar (replicateM sz norm) StdRandom
	return $ learnee odot (fromList bs)

activator ∷ Num a ⇒ Unary a → Learnee a a
activator f = computee (unary f)

norm ∷ (Distribution Normal a, Fractional a) ⇒ RVar a
norm = normal (-1.0) 1.0
