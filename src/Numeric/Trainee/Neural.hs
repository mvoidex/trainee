{-# LANGUAGE GADTs, FlexibleContexts, RankNTypes #-}

module Numeric.Trainee.Neural (
	LayerBuild,
	Layer,
	NetBuild(..), Net,
	net, input, (⭃),
	fc, conv, conv2,

	sigma, relu, softplus,
	summator, biaser, activator, convolver, convolver2,
	normVar,

	module Numeric.Trainee.Types,
	module Numeric.Trainee.Learnee
	) where

import Prelude hiding ((.), id)

import Control.Category
import Control.Monad (replicateM, liftM)
import Data.Random
import Numeric.LinearAlgebra hiding (conv, conv2)

import Numeric.Trainee.Types
import Numeric.Trainee.Gradee
import Numeric.Trainee.Learnee

type LayerBuild m a = Int → m (Layer a, Int)

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
	(l', out') ← l (buildOut n')
	return $ n' {
		buildOut = out',
		buildNet = buildNet n' ⇉ l' }

-- | Fully connected layer
fc ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Unary (Vector a) → Int → LayerBuild m a
fc f outputs inputs = do
	s ← summator outputs inputs
	b ← biaser outputs
	return (s ⇉ b ⇉ activator f, outputs)

-- | Convolution layer 1-d
conv ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Unary (Vector a) → Int → LayerBuild m a
conv f w inputs = do
	c ← convolver w
	b ← do
		bs ← runRVar normVar StdRandom
		return $ learnee biasVec bs
	return (c ⇉ b ⇉ activator f, inputs - w + 1)

-- | Convolution layer 2-d
conv2 ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Unary (Vector a) → Int → (Int, Int) → LayerBuild m a
conv2 f cols' (w, h) inputs = do
	c ← convolver2 w h
	b ← do
		bs ← runRVar normVar StdRandom
		return $ learnee biasMat bs
	let
		rows' = inputs `div` cols'
	return (computee (reshapeVec cols') ⇉ c ⇉ b ⇉ computee flattenMat ⇉ activator f, (rows' - h + 1) * (cols' - w + 1))

sigma ∷ Floating a ⇒ a → a
sigma t = 1 / (1 + exp (negate t))

relu ∷ (Fractional a) ⇒ a → a
relu t = 0.5 * (1 + signum t) * t

softplus ∷ Floating a ⇒ a → a
softplus t = log (1 + exp t)

summator ∷ (MonadRandom m, Distribution Normal a, Fractional a, Numeric a, Num (Vector a), Parametric a) ⇒ Int → Int → m (Layer a)
summator outputs inputs = do
	ws ← runRVar (replicateM (inputs * outputs) normVar) StdRandom
	return $ learnee matVec ((outputs >< inputs) ws)

biaser ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Int → m (Layer a)
biaser sz = do
	bs ← runRVar (replicateM sz normVar) StdRandom
	return $ learnee odot (fromList bs)

activator ∷ Num a ⇒ Unary a → Learnee a a
activator f = computee (unary f)

convolver ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Int → m (Layer a)
convolver w = do
	ws ← runRVar (replicateM w normVar) StdRandom
	return $ learnee corrVec (fromList ws)

convolver2 ∷ (MonadRandom m, Distribution Normal a, Fractional a, Numeric a, Num (Vector a), Parametric a) ⇒ Int → Int → m (Learnee (Matrix a) (Matrix a))
convolver2 w h = do
	ws ← runRVar (replicateM (w * h) normVar) StdRandom
	return $ learnee corrMat ((w >< h) ws)

normVar ∷ (Distribution Normal a, Fractional a) ⇒ RVar a
normVar = normal 0.0 0.25
