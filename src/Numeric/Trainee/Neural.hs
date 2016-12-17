{-# LANGUAGE GADTs, FlexibleContexts, RankNTypes #-}

module Numeric.Trainee.Neural (
	Layer, layerIn, layerOut,
	NetBuild(..), Net(..),
	net, input, (⭃),
	fc,

	sigma, relu,
	summator, activator,
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
import Numeric.Trainee.Gradee (matVec, swap, unary, Unary)
import Numeric.Trainee.Learnee

type Layer a = Learnee (Matrix a) (Vector a) (Vector a)

layerIn ∷ (Container Vector a, Num a) ⇒ Layer a → Int
layerIn = snd ∘ size ∘ _params

layerOut ∷ (Container Vector a, Num a) ⇒ Layer a → Int
layerOut = fst ∘ size ∘ _params

data NetBuild w a = NetBuild {
	buildOut ∷ Int,
	buildNet ∷ Learnee w (Vector a) (Vector a) }

data Net a where
	Net ∷ (Show w, Params w) ⇒ Learnee w (Vector a) (Vector a) → Net a

net ∷ (Show w, Params w) ⇒ RVar (NetBuild w a) → IO (Net a)
net act = liftM (Net ∘ buildNet) $ runRVar act StdRandom

input ∷ Monad m ⇒ Int → m (NetBuild () a)
input i = return $ NetBuild i (computee id)

(⭃) ∷ (Combine w (Matrix a), Monad m, Num a, Container Vector a) ⇒ m (NetBuild w a) → (Int → m (Layer a)) → m (NetBuild (CombineResult w (Matrix a)) a)
n ⭃ l = do
	n' ← n
	l' ← l (buildOut n')
	return $ n' {
		buildOut = layerOut l',
		buildNet = buildNet n' ⇉ l' }

-- | Fully connected layer
fc ∷ (MonadRandom m, Distribution Normal a, Num (Vector a), Fractional a, Numeric a) ⇒ Unary (Vector a) → Int → Int → m (Layer a)
fc f outputs inputs = do
	s ← summator outputs inputs
	return $ s ⇉ activator f

sigma ∷ Floating a ⇒ a → a
sigma t = 1 / (1 + exp (negate t))

relu ∷ (Ord a, Num a) ⇒ a → a
relu = max 0

summator ∷ (MonadRandom m, Distribution Normal a, Fractional a, Numeric a) ⇒ Int → Int → m (Layer a)
summator outputs inputs = do
	ws ← runRVar (replicateM (inputs * outputs) norm) StdRandom
	return $ learnee (matVec . swap) ((inputs >< outputs) ws)

activator ∷ Num a ⇒ Unary a → Computee a a
activator f = computee (unary f)

norm ∷ (Distribution Normal a, Fractional a) ⇒ RVar a
norm = normal (-1.0) 1.0
